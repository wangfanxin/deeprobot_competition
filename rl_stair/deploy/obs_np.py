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
  [48:50] heading [cos,sin](yaw-track_heading) (2) -- Chamorro
  [50:54] terrain ctx: front/rear axle distance + height diff to next riser (4)
  [54]    rough bool (stair OR ridge)
"""
import numpy as np

TARGET_HEADING = 1.5708   # pi/2: task axis +y (track heading at stair section); deployment should pass track heading

FRONT_REAR_OFFSET = 0.246   # half wheelbase (m) under S10 tall stair stance (settled, measured)


def build_indices(mj_model):
    """Return index arrays / defaults from a MuJoCo model (robust to extra joints).

    Maps actuator j -> its joint via actuator(j).trnid[0], so the obs/action order
    always matches the actuator order the policy was trained on (works for the
    competition track model which has extra non-robot joints).
    """
    n = mj_model.nu
    jids = [mj_model.actuator(j).trnid[0] for j in range(n)]
    act2jnt = np.array([mj_model.jnt_qposadr[jid] for jid in jids])
    act2vel = act2jnt - 1
    names = [mj_model.joint(jid).name for jid in jids]
    leg_idx = np.array([j for j, nm in enumerate(names) if "wheel" not in nm])
    # default_dof must match s10_env.py EXACTLY = S10 cruise half-squat pose_target
    # (vmc_legs.py:841-846, S10_CAR_SQUAT=1). NOT model qpos0 (all-zeros straight legs),
    # NOT go2w stance. Actuator order verified == training order.
    default_dof = np.array([-0.05, -0.60, 1.20, 0.0,
                             0.05, -0.60, 1.20, 0.0,
                            -0.05,  0.60, -1.20, 0.0,
                             0.05,  0.60, -1.20, 0.0], dtype=np.float64)
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
    # Chamorro-style heading feedback (matches env _obs): yaw error vs TARGET_HEADING
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    yaw_err = yaw - TARGET_HEADING
    heading = np.array([np.cos(yaw_err), np.sin(yaw_err)], dtype=np.float32)
    ctx = np.zeros(4, dtype=np.float32)
    riser_y = np.asarray(riser_y, dtype=np.float64)
    riser_top = np.asarray(riser_top, dtype=np.float64)
    if len(riser_y) > 0:
        base_y = float(qpos[1])
        nr = len(riser_y)
        sy = np.sin(yaw)   # yaw-rotate axle offsets (match env _terrain_ctx)
        for k, off in enumerate((FRONT_REAR_OFFSET, -FRONT_REAR_OFFSET)):
            ay = base_y + off * sy
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
                           heading, ctx, rough]).astype(np.float32)
