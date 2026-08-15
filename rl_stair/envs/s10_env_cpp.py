import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""s10_env_cpp.py: C++ MuJoCo (CPU) training backend for RL-stair (USER-APPROVED 2026-08-15).

Purpose: MJX<->C++ wheel-drive 2.7x gap (doc 3.36) makes MJX-trained policies
unpredictable in the official C++ sim. Training directly in C++ MuJoCo learns the
REAL contact/solver behavior -> official-env (S10_track.xml) success toward 95%.

Same 55-dim obs / PD controller as s10_env.py (single source: deploy/obs_np.py).
Competition stair geometry as LIGHT box terrain (fast mj_step); the official mesh
S10_track.xml is used only for ACCEPTANCE, not training.
"""
import numpy as np
import mujoco
from rl_stair.deploy.obs_np import build_indices, compute_obs_np

KP_LEG, KD_LEG, KP_WHEEL = 50.0, 1.0, 2.0
TORQ_LEG, TORQ_WHEEL = 48.0, 13.5
ACTION_SCALE, VEL_SCALE = 0.7, 24.0
GROUND = 0.48
RISERS_Y = np.array([37.90, 38.375, 38.775, 39.225, 39.625, 40.025])
TOPS = np.array([0.54, 0.67, 0.79, 0.92, 1.04, 1.17])
TREADS = [0.475, 0.40, 0.45, 0.40, 0.40, 0.40]
REACH_Y = 41.271


ROBOT_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "src", "S10_sdk_deploy", "S10_description", "s10_mjcf", "mjcf", "S10.xml")


def build_terrain_xml(robot_xml=ROBOT_XML):
    """Official S10 robot (cylinder wheels, C++ solver) + light box competition stairs."""
    xml = open(robot_xml, encoding="utf-8").read()
    # fix meshdir to absolute (from_xml_string resolves relative to cwd)
    _mesh_abs = os.path.join(os.path.dirname(robot_xml), "..", "meshes")
    xml = xml.replace('meshdir="../meshes/"', 'meshdir="%s"' % os.path.abspath(_mesh_abs))
    # remove stock floor so the track terrain is the only support
    xml = __import__("re").sub(r"<geom name=['\"]floor['\"][^>]*/>", "", xml)
    parts = []
    parts.append('<geom name="ground" type="plane" size="20 60 0.02" pos="0 20 0" friction="1"/>')
    parts.append('<geom name="approach" type="box" size="1.5 %.4f %.4f" pos="0 %.4f %.4f" friction="1"/>'
                 % ((37.9 - 31.5) / 2, GROUND / 2, (31.5 + 37.9) / 2, GROUND / 2))
    for i in range(6):
        y0 = RISERS_Y[i]; tr = TREADS[i]; top = TOPS[i]
        parts.append('<geom name="stair%d" type="box" size="1.5 %.4f %.4f" pos="0 %.4f %.4f" friction="0.8"/>'
                     % (i, tr / 2, top / 2, y0 + tr / 2, top / 2))
    last_end = float(RISERS_Y[-1] + TREADS[-1])
    parts.append('<geom name="top" type="box" size="1.5 5.0 %.4f" pos="0 %.4f %.4f" friction="0.8"/>'
                 % (float(TOPS[-1]) / 2, last_end + 5.0, float(TOPS[-1]) / 2))
    terrain = "".join(parts)
    assert "</worldbody>" in xml
    return xml.replace("</worldbody>", terrain + "</worldbody>")


OFFICIAL_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "src", "S10_sdk_deploy", "S10_description", "s10_mjcf", "mjcf", "S10_track.xml")


class S10EnvCPP:
    def __init__(self, num_envs=64, xml=None, official=False):
        self.n = num_envs
        self.official = official
        if official:
            self.model = mujoco.MjModel.from_xml_path(OFFICIAL_XML)
            self.xml = OFFICIAL_XML
        else:
            self.xml = xml or build_terrain_xml()
            self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.model.opt.timestep = 0.005   # official 200Hz sim
        self.model.opt.cone = 0
        self.model.opt.jacobian = 0
        self.idx = build_indices(self.model)
        self.default_dof = self.idx["default_dof"]
        _names = [self.model.joint(self.model.actuator(j).trnid[0]).name for j in range(self.model.nu)]
        self.idx["wheel_idx"] = np.array([j for j, nm in enumerate(_names) if "wheel" in nm])
        self.nu = self.model.nu
        self.obs_dim = 55
        self.action_size = 16
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.last_action = np.zeros((num_envs, self.nu), np.float32)
        self.cmd = np.tile(np.array([1.5, 0.0], np.float32), (num_envs, 1))
        self.ep_len = np.zeros(num_envs, np.int32)

    def _settle(self, k):
        """PD-hold settle: robot lands on the approach before driving (official pattern)."""
        d = self.data[k]
        for _ in range(500):
            q = d.qpos[self.idx["act2jnt"]] - self.default_dof
            qd = d.qvel[self.idx["act2vel"]]
            tau = np.zeros(self.nu)
            lt = np.clip(KP_LEG * (0.0 - q) - KD_LEG * qd, -TORQ_LEG, TORQ_LEG)
            tau[self.idx["leg_idx"]] = lt[self.idx["leg_idx"]]
            d.ctrl[:] = tau
            mujoco.mj_step(self.model, d)
            if d.qpos[2] < 0.25:
                break

    def reset(self, seeds=None):
        for k in range(self.n):
            rng = np.random.default_rng(1000 + k if seeds is None else seeds[k])
            d = self.data[k]
            x = float(rng.uniform(-0.1, 0.1)); yaw = 1.5708 + float(rng.uniform(-0.05, 0.05))
            d.qpos[:] = 0; d.qvel[:] = 0
            d.qpos[0:3] = [x, 32.0, 0.9]
            d.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
            for i, j in enumerate(self.idx["act2jnt"]): d.qpos[j] = self.default_dof[i]
            mujoco.mj_forward(self.model, d)
            self._settle(k)
        self.last_action[:] = 0
        self.ep_len[:] = 0
        return self._obs_all()

    def step(self, actions):
        n = self.n
        rew = np.zeros(n); done = np.zeros(n, bool); succ = np.zeros(n, bool)
        for k in range(n):
            d = self.data[k]
            a = np.clip(actions[k], -1, 1)
            q = d.qpos[self.idx["act2jnt"]] - self.default_dof
            qd = d.qvel[self.idx["act2vel"]]
            tau = np.zeros(self.nu)
            lt = np.clip(KP_LEG*(a*ACTION_SCALE - q) - KD_LEG*qd, -TORQ_LEG, TORQ_LEG)
            tau[self.idx["leg_idx"]] = lt[self.idx["leg_idx"]]
            wt = np.clip(KP_WHEEL*(a*VEL_SCALE - qd), -TORQ_WHEEL, TORQ_WHEEL)
            tau[self.idx["wheel_idx"]] = wt[self.idx["wheel_idx"]]
            d.ctrl[:] = tau
            for _ in range(4):          # 4 substeps = 50Hz policy (matches training)
                mujoco.mj_step(self.model, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    # NaN contact explosion (cylinder wheel on riser edge): force reset
                    break
            self.last_action[k] = a
            by, bz = d.qpos[1], d.qpos[2]
            self.ep_len[k] += 1
            rew[k] += 2.0 * max(0.0, float(d.qvel[1])) * 0.005
            if by > REACH_Y and bz > 0.15:
                succ[k] = True; rew[k] += 10.0
            if bz < 0.15 or self.ep_len[k] > 2000:
                done[k] = True
        return self._obs_all(), rew, done, succ

    def _obs_all(self):
        obs = np.zeros((self.n, self.obs_dim), np.float32)
        for k in range(self.n):
            d = self.data[k]
            obs[k] = compute_obs_np(d.qpos, d.qvel, self.idx, self.last_action[k],
                                    self.cmd[k], RISERS_Y, TOPS)
        return obs


if __name__ == "__main__":
    import time
    env = S10EnvCPP(num_envs=8)
    obs = env.reset()
    print("obs", obs.shape, obs.dtype)
    t0 = time.time()
    for _ in range(50):
        obs, rew, done, succ = env.step(np.zeros((8, 16)))
    dt = (time.time() - t0) / 50
    print("8 env step %.1f ms -> 64 env est %.0f ms/step, 24 steps/update %.2fs"
          % (dt*1000, dt*8*1000, dt*8*24))
    print("base z:", [round(float(env.data[k].qpos[2]), 3) for k in range(8)])
