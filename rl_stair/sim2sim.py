"""Sim2sim validation: competition S10_track.xml + exported policy (wp6->7 stairs).

Read-only harness (CPU mujoco, no training-GPU use, no track/DiAL edits).
Runs the exported TorchScript policy at 50Hz on the real track, spawned before
the wp6->7 stairs (STAIR_RISERS/TOPS from DiAL), reports risers crossed / fall /
progress. obs encoding via deploy/obs_np.py (verified == MJX training env).

Usage:
  python rl_stair/sim2sim.py --ckpt rl_stair/deploy/policy.pt \
      [--x -14.4 --y 37.0 --yaw 0 --vx 1.5 --steps 1500]
"""
import os, sys, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco, torch
from rl_stair.deploy.obs_np import compute_obs_np, build_indices

TRACK = ("/home/wfx/DR_competition/0810new/deeprobot_competition/"
         "src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10_track.xml")
RISERS = np.array([37.90, 38.375, 38.775, 39.225, 39.625, 40.025])
TOPS = np.array([0.54, 0.67, 0.79, 0.92, 1.04, 1.17])
GROUND = 0.48
KP_LEG, KD_LEG, KP_WHEEL = 50.0, 1.0, 2.0
ACTION_SCALE, VEL_SCALE = 0.25, 10.0
TORQ_LEG, TORQ_WHEEL = 48.0, 13.5
DECIMATION = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--x", type=float, default=-14.4)
    ap.add_argument("--y", type=float, default=37.0)
    ap.add_argument("--yaw", type=float, default=1.5708)   # wp6->7 track heading ~88.85 deg (+y); training reset body+x=+y. old default 0 drove perpendicular to stairs
    ap.add_argument("--vx", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=1500)   # policy steps (50Hz)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(TRACK)
    d = mujoco.MjData(m)
    idx = build_indices(m)
    act2jnt, act2vel = idx["act2jnt"], idx["act2vel"]
    leg_idx, wheel_idx = idx["leg_idx"], np.array([j for j in range(m.nu) if j not in idx["leg_idx"]])
    default_dof = idx["default_dof"]

    # spawn before stairs, facing along track (+y), base z = ground + stand height
    stand_z = 0.354   # == S10RLEnv._compute_stand (training env), same robot geometry
    cy, sy = math.cos(args.yaw / 2), math.sin(args.yaw / 2)
    d.qpos[0:3] = [args.x, args.y, GROUND + stand_z]
    d.qpos[3:7] = [cy, 0.0, 0.0, sy]
    mujoco.mj_forward(m, d)

    policy = torch.jit.load(args.ckpt)
    policy.eval()
    last_action = np.zeros(m.nu, dtype=np.float32)
    cmd = np.array([args.vx, 0.0], dtype=np.float32)

    def step_policy(action):
        nonlocal d, last_action
        q = d.qpos[act2jnt] - default_dof
        qd = d.qvel[act2vel]
        tau = np.zeros(m.nu, dtype=np.float64)
        q_target = action * ACTION_SCALE
        leg_tau = np.clip(KP_LEG * (q_target - q) - KD_LEG * qd, -TORQ_LEG, TORQ_LEG)
        tau[leg_idx] = leg_tau[leg_idx]
        vel_ref = action * VEL_SCALE
        w_tau = np.clip(KP_WHEEL * (vel_ref - qd), -TORQ_WHEEL, TORQ_WHEEL)
        tau[wheel_idx] = w_tau[wheel_idx]
        for _ in range(DECIMATION):
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        last_action = action.copy()

    max_prog = 0.0
    risers_crossed = 0
    fell = False
    for i in range(args.steps):
        obs = compute_obs_np(d.qpos, d.qvel, idx, last_action, cmd, RISERS, TOPS)
        with torch.no_grad():
            a = policy(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
        step_policy(np.clip(a, -1.0, 1.0))
        base_y, base_z = float(d.qpos[1]), float(d.qpos[2])
        prog = max(0.0, base_y - args.y)
        max_prog = max(max_prog, prog)
        risers_crossed = max(risers_crossed, int(np.sum(RISERS < base_y)))
        if base_z < GROUND + 0.15:
            fell = True
            print("[%04d] FALL at y=%.2f (z=%.3f) risers=%d" % (i, base_y, base_z, risers_crossed))
            break
        if base_y > RISERS[-1] + 0.5 and i % 100 == 0:
            print("[%04d] y=%.2f risers=%d (climbed past last riser)" % (i, base_y, risers_crossed))

    print("RESULT y_final=%.2f max_prog=%.2fm risers_crossed=%d/6 fell=%s steps=%d" % (
        float(d.qpos[1]), max_prog, risers_crossed, fell, min(i + 1, args.steps)))


if __name__ == "__main__":
    main()
