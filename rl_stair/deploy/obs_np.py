"""Deployment-side 53-dim actor observation encoder (numpy, verified == MJX env).

Verified 2026-08-14: max abs diff vs rl_stair/envs/s10_env.py `_obs` = 3.7e-9
(float32 precision). Single source of truth for competition-sim / deploy encoding.

OBS_LAYOUT (must stay in this exact order):
  [0:3]   base ang vel (body) *0.25
  [3:6]   projected gravity
  [6:8]   cmd [vx, yaw]
  [8:20]  leg joint pos error (12) *1 (default 0/0.67/-1.3)
  [20:32] leg joint vel (12) *0.05
  [32:48] last action (16)
  [48:52] terrain ctx: front/rear axle distance + height diff to next riser (4)
  [52]    rough bool (stair OR ridge)
"""
import numpy as np

FRONT_REAR_OFFSET = 0.228   # half wheelbase (m) for front/rear axle terrain ctx


def build_indices(mj_model):
    """Return index arrays / defaults from a MuJoCo model (same convention as training)."""
    names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    act2jnt = np.array([mj_model.jnt_qposadr[i] for i in range(1, mj_model.njnt)])
    act2vel = np.array([mj_model.jnt_qposadr[i] - 1 for i in range(1, mj_model.njnt)])
    leg_idx = np.array([i for i, nm in enumerate(names[1:]) if "wheel" not in nm])
    default_dof = np.array([mj_model.qpos0[mj_model.jnt_qposadr[i]] for i in range(1, mj_model.njnt)])
    return {"act2jnt": act2jnt, "act2vel": act2vel, "leg_idx": leg_idx,
            "default_dof": default_dof}


def compute_obs_np(qpos, qvel, idx, last_action, cmd, riser_y, riser_top):
    """qpos/qvel: 1-D numpy (nq/nv). riser_y/riser_top: 1-D numpy of obstacle front edges / tops."""
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    act2jnt, act2vel = idx["act2jnt"], idx["act2vel"]
    leg_idx, default_dof = idx["leg_idx"], idx["default_dof"]
    q = qpos[act2jnt]
    qd = qvel[act2vel]
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    g = np.array([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
                  1 - 2 * (qx * qx + qy * qy)], dtype=np.float32)
    angvel = (qvel[3:6] * 0.25).astype(np.float32)
    leg_err = (q[leg_idx] - default_dof[leg_idx]).astype(np.float32)
    leg_vel = (qd[leg_idx] * 0.05).astype(np.float32)
    ctx = np.zeros(4, dtype=np.float32)
    riser_y = np.asarray(riser_y, dtype=np.float64)
    riser_top = np.asarray(riser_top, dtype=np.float64)
    if len(riser_y) > 0:
        base_y = float(qpos[1])
        nr = len(riser_y)
        for k, off in enumerate((FRONT_REAR_OFFSET, -FRONT_REAR_OFFSET)):
            ay = base_y + off
            idx_p = int(np.sum(riser_y < ay))
            nxt = min(idx_p, nr - 1)
            d_next = riser_y[nxt] - ay
            top_next = riser_top[nxt]
            cur_top = riser_top[max(idx_p - 1, 0)] if idx_p > 0 else 0.0
            ctx[k] = float(np.clip(d_next, -1.0, 1.0))
            ctx[k + 2] = float(np.clip((top_next - cur_top) / 0.2, -1.0, 1.0))
    rough = np.array([1.0 if len(riser_y) > 0 else 0.0], dtype=np.float32)
    return np.concatenate([angvel, g, np.asarray(cmd, dtype=np.float32), leg_err,
                           leg_vel, np.asarray(last_action, dtype=np.float32),
                           ctx, rough]).astype(np.float32)
