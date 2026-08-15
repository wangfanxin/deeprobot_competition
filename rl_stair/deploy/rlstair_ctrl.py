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
_DEFAULT_POLICY = os.path.join(_HERE, "policy.pt")

KP_LEG, KD_LEG = 50.0, 1.0
KP_WHEEL = 2.0
TORQ_LEG, TORQ_WHEEL = 48.0, 13.5
ACTION_SCALE, VEL_SCALE = 0.7, 24.0


class RLStairCtrl:
    def __init__(self, m, policy_path=_DEFAULT_POLICY, vx=1.5):
        from rl_stair.deploy.obs_np import build_indices
        from rl_stair.ppo import PPO, PPOCfg
        self.idx = build_indices(m)
        self._names = [m.joint(m.actuator(j).trnid[0]).name for j in range(m.nu)]
        self.leg_idx = np.array([j for j, nm in enumerate(self._names) if "wheel" not in nm])
        self.wheel_idx = np.array([j for j, nm in enumerate(self._names) if "wheel" in nm])
        self.default_dof = self.idx["default_dof"]
        self.ppo = PPO(55, 72, 16, PPOCfg(num_envs=1), "cpu")
        if not os.path.exists(policy_path):
            raise FileNotFoundError("policy not found: " + policy_path)
        self.ppo.load(policy_path)
        self.ppo.actor.eval()
        self.last_action = np.zeros(m.nu, dtype=np.float32)
        self.cmd = np.array([vx, 0.0], dtype=np.float32)
        self.riser_y = np.array([], dtype=np.float64)
        self.riser_top = np.array([], dtype=np.float64)

    def set_risers(self, riser_y, riser_top):
        self.riser_y = np.asarray(riser_y, dtype=np.float64)
        self.riser_top = np.asarray(riser_top, dtype=np.float64)

    def set_cmd(self, vx):
        self.cmd[0] = float(vx)

    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel, cmd, terr, DT):
        from rl_stair.deploy.obs_np import compute_obs_np
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        if cmd is not None:
            self.cmd[0] = float(cmd.get("vx", self.cmd[0]))
        obs = compute_obs_np(qpos, qvel, self.idx, self.last_action, self.cmd,
                             self.riser_y, self.riser_top)
        with torch.no_grad():
            a = self.ppo.actor.act(
                torch.as_tensor(obs).unsqueeze(0), noiseless=True).squeeze(0).numpy()
        a = np.clip(a, -1.0, 1.0)
        q = qpos[self.idx["act2jnt"]] - self.default_dof
        qd = qvel[self.idx["act2vel"]]
        tau = np.zeros(self.ppo.action_size if hasattr(self.ppo, "action_size") else 16,
                       dtype=np.float64)
        leg_tau = np.clip(KP_LEG * (a * ACTION_SCALE - q) - KD_LEG * qd,
                          -TORQ_LEG, TORQ_LEG)
        tau[self.leg_idx] = leg_tau[self.leg_idx]
        wt = np.clip(KP_WHEEL * (a * VEL_SCALE - qd), -TORQ_WHEEL, TORQ_WHEEL)
        tau[self.wheel_idx] = wt[self.wheel_idx]
        self.last_action = a.copy()
        return tau
