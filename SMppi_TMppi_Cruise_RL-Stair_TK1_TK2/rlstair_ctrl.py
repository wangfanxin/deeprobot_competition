"""rlstair_ctrl.py: deployment RL-stair controller (S10_VMC_MODE=rlstair).

Mirrors the training + sim2sim_exact control exactly:
  obs_np.compute_obs_np (55-dim, verified == MJX env) -> policy forward -> 
  leg position PD (kp50/kd1, clip48) + wheel velocity (kp2*vel24, clip13.5).
Policy: rl_stair/deploy/policy.pt (exported actor, 55->16, tanh).

VMC interface: compute_tau(qpos, qvel, wheel_xyz, wheel_vel, cmd, terr, DT) -> tau(16)
with risers set externally via set_risers(riser_y, riser_top) (known map or lidar).
CPU-only (deploy on Orin / competition sim).
"""
import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_POLICY = os.path.join(_HERE, "policy.pt")  # 本文件夹自带 policy
# A/B eval: override the policy path per-run (default deploy/policy.pt).
# 2026-08-16: used to evaluate training checkpoints through the INTEGRATED flow
# (the only valid real-mesh test - standalone RL falls through the thin mesh shell).
_DEFAULT_POLICY = os.environ.get("S10_RL_POLICY", _DEFAULT_POLICY)

KP_LEG, KD_LEG = 50.0, 1.0
KP_WHEEL = 2.0
TORQ_LEG, TORQ_WHEEL = 48.0, 13.5
ACTION_SCALE, VEL_SCALE = 0.7, 24.0
# BUGFIX 2026-08-16 (GOAL #2): training runs the policy at 50Hz with 200Hz sim
# (s10_env decimation=4). The integrated controller called the policy EVERY 200Hz
# control step -> gait/action dynamics ran 4x faster than trained -> mid-stair
# instability. Run the policy every DECIMATION-th call (50Hz), hold the action
# (zero-order hold) for the tau PD between policy steps.
DECIMATION = 4


class RLStairCtrl:
    def __init__(self, m, policy_path=_DEFAULT_POLICY, vx=1.5):
        from rlstair_obs import build_indices
        self.idx = build_indices(m)
        self._names = [m.joint(m.actuator(j).trnid[0]).name for j in range(m.nu)]
        self.leg_idx = np.array([j for j, nm in enumerate(self._names) if "wheel" not in nm])
        self.wheel_idx = np.array([j for j, nm in enumerate(self._names) if "wheel" in nm])
        self.default_dof = self.idx["default_dof"]
        if not os.path.exists(policy_path):
            raise FileNotFoundError("policy not found: " + policy_path)
        # BUGFIX 2026-08-15 23:35: policy.pt is a JIT TorchScript archive (export.py
        # torch.jit.save); torch.load(weights_only=True) rejects it. Load the JIT model
        # directly and call forward() = tanh(MLP(obs)) -> action in [-1,1].
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()
        self._act_dim = self.policy(torch.zeros(1, 55)).shape[-1]
        self.last_action = np.zeros(int(self._act_dim), dtype=np.float32)
        self.cmd = np.array([vx, 0.0], dtype=np.float32)
        # 2026-08-18: RL 航向目标（rad），默认 pi/2 = 赛道楼梯方向；
        # TK1 交接时经 set_heading() 设为 riser 爬升方向。
        self.target_heading = float(os.environ.get("S10_RL_HEADING", "1.5708"))
        self.riser_xy = np.zeros((0, 2), dtype=np.float64)
        self.riser_top = np.array([], dtype=np.float64)
        self.climb_axis = np.array([np.cos(self.target_heading),
                                    np.sin(self.target_heading)], dtype=np.float64)
        self.climb_origin = np.zeros(2, dtype=np.float64)
        self._pol_step = 0   # 200Hz call counter; policy runs every DECIMATION
        # USER-DIRECTED 2026-08-16 (GOAL #2): handoff warm-start. Training resets with
        # the legs AT default_dof (after the env's stand-PD warmup); the integrated
        # flow hands off from CRUISE half-squat (hipy=-1.10/knee=1.90) so the first RL
        # obs leg_err is HUGE (out-of-distribution) -> the policy collapses/misbehaves
        # on the stairs. Hold the default stance with the stand PD for the first
        # S10_RL_WARMUP control steps (~1s), then run the policy from a matching pose.
        self._warm = int(os.environ.get("S10_RL_WARMUP", "200"))

    def set_risers(self, riser_xy, riser_top, heading=None, origin=None):
        """注入 lidar 在线检测的 riser 世界坐标与台面高。

        riser_xy: (N,2) world xy；riser_top: (N,)。所有距离在
        climb_axis（楼梯爬升方向）上投影，不依赖世界 y 轴或已知地图表。
        """
        self.riser_xy = np.asarray(riser_xy, dtype=np.float64).reshape(-1, 2)
        self.riser_top = np.asarray(riser_top, dtype=np.float64).reshape(-1)
        if heading is not None:
            self.climb_axis = np.array([np.cos(float(heading)),
                                        np.sin(float(heading))], dtype=np.float64)
        if origin is None:
            self.climb_origin = self.riser_xy[0].copy() if len(self.riser_xy) else np.zeros(2)
        else:
            self.climb_origin = np.asarray(origin, dtype=np.float64).reshape(2)

    def set_cmd(self, vx):
        self.cmd[0] = float(vx)

    def set_heading(self, heading):
        """2026-08-18 (坐标系转换): RL 航向目标，默认 pi/2 = 赛道楼梯方向。"""
        self.target_heading = float(heading)

    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel, cmd, terr, DT):
        from rlstair_obs import compute_obs_np
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        # USER-DIRECTED 2026-08-16: stair section does NOT track the nav ref_v;
        # the RL policy controls its own speed (self.cmd set by set_cmd). Ignore
        # the nav vx passed in so the policy's learned speed profile is used.
        if cmd is not None:
            pass  # RL self-speed only
        self._pol_step += 1
        if self._pol_step <= self._warm:
            # USER-DIRECTED 2026-08-16: brief PD takeover for the cruise half-squat ->
            # RL tall-stance transition. Hold default_dof (stand PD, same law as the
            # training warmup), wheels FREE. The robot must arrive SLOW (cruise decels
            # to ~1.5 by the handoff) so the free-wheel warm coasts little; per-wheel
            # braking to 0 was REVERTED (verified 2026-08-16 handoff22: asymmetric
            # brake torques amplify yaw error -> robot spun/rolled at y~35.5).
            q = qpos[self.idx["act2jnt"]] - self.default_dof
            qd = qvel[self.idx["act2vel"]]
            tau = np.zeros(int(self._act_dim), dtype=np.float64)
            lt = np.clip(KP_LEG * (0.0 - q) - KD_LEG * qd, -TORQ_LEG, TORQ_LEG)
            tau[self.leg_idx] = lt[self.leg_idx]
            return tau
        # BUGFIX 2026-08-16: decimation must be counted from AFTER the warm-start,
        # otherwise a warm length where (warm+1) % DECIMATION != 1 leaves _action
        # unset on the first post-warm step (crash). Force a policy step there.
        if (self._pol_step - self._warm - 1) % DECIMATION == 0:
            # policy step (50Hz): fresh action from current obs
            obs = compute_obs_np(qpos, qvel, self.idx, self.last_action, self.cmd,
                                 self.riser_xy, self.riser_top,
                                 target_heading=self.target_heading,
                                 climb_axis=self.climb_axis,
                                 climb_origin=self.climb_origin)
            with torch.no_grad():
                a = self.policy(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            self._action = np.clip(a, -1.0, 1.0)
            self.last_action = self._action.copy()
        a = self._action
        if os.environ.get('S10_RL_DEBUG', '0') == '1':
            print('[RLDBG] a_leg=', np.round(a[self.leg_idx], 3),
                  'a_wheel=', np.round(a[self.wheel_idx], 3),
                  'qd_wheel=', np.round(qvel[self.idx["act2vel"]][self.wheel_idx], 3),
                  flush=True)
        q = qpos[self.idx["act2jnt"]] - self.default_dof
        qd = qvel[self.idx["act2vel"]]
        tau = np.zeros(int(self._act_dim), dtype=np.float64)
        leg_tau = np.clip(KP_LEG * (a * ACTION_SCALE - q) - KD_LEG * qd,
                          -TORQ_LEG, TORQ_LEG)
        tau[self.leg_idx] = leg_tau[self.leg_idx]
        wt = np.clip(KP_WHEEL * (a * VEL_SCALE - qd), -TORQ_WHEEL, TORQ_WHEEL)
        tau[self.wheel_idx] = wt[self.wheel_idx]
        return tau
