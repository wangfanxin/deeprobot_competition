"""Deployment-side 55-dim actor observation encoder (numpy, verified == MJX env).

Verified 2026-08-14: max abs diff vs rl_stair/envs/s10_env.py `_obs` = 3.7e-9
(float32 precision). Single source of truth for competition-sim / deploy encoding.

OBS_LAYOUT (must stay in this exact order):
  [0:3]   base ang vel (body) *0.25
  [3:6]   projected gravity
  [6:8]   cmd [vx, yaw]
  [8:20]  leg joint pos error (12) *1 (default S10 tall stance [∓0.05,∓0.60,±1.20])
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
# default_dof must match s10_env.py EXACTLY = S10 tall stair stance (USER-DIRECTED 2026-08-14)
# NOT model qpos0 (all-zeros straight legs); NOT go2w stance; NOT cruise half-squat.
    # NOT go2w stance. Actuator order verified == training order.
    default_dof = np.array([-0.05, -0.60, 1.20, 0.0,
                             0.05, -0.60, 1.20, 0.0,
                            -0.05,  0.60, -1.20, 0.0,
                             0.05,  0.60, -1.20, 0.0], dtype=np.float64)
    return {"act2jnt": act2jnt, "act2vel": act2vel, "leg_idx": leg_idx,
            "default_dof": default_dof}


def compute_obs_np(qpos, qvel, idx, last_action, cmd, riser_xy, riser_top,
                   target_heading=TARGET_HEADING, climb_axis=None,
                   climb_origin=None):
    """qpos/qvel: 1-D numpy (nq/nv).

    riser_xy: (N,2) world xy of riser front edges（lidar 在线检测）;
              为兼容旧调用，1-D 输入按 climb_axis 方向标量处理。
    riser_top: (N,) top height.
    climb_axis/climb_origin: 楼梯爬升方向单位向量与原点；缺省为世界 +y 与原点
    （等价训练环境）。所有 riser 距离在 climb_axis 上投影，消除 y 轴硬编码。
    """
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
    yaw_err = yaw - target_heading
    heading = np.array([np.cos(yaw_err), np.sin(yaw_err)], dtype=np.float32)
    ctx = np.zeros(4, dtype=np.float32)
    riser_top = np.asarray(riser_top, dtype=np.float64)
    riser_xy = np.asarray(riser_xy, dtype=np.float64)
    if climb_axis is None:
        climb_axis = np.array([0.0, 1.0], dtype=np.float64)
    else:
        climb_axis = np.asarray(climb_axis, dtype=np.float64)
    if climb_origin is None:
        climb_origin = np.zeros(2, dtype=np.float64)
    else:
        climb_origin = np.asarray(climb_origin, dtype=np.float64)
    # riser 标量坐标 = 沿爬升方向的投影；兼容旧的 1-D 标量输入
    if riser_xy.ndim == 1:
        riser_t = riser_xy.astype(np.float64)
    else:
        riser_t = (riser_xy.reshape(-1, 2) - climb_origin[None, :]) @ climb_axis
    order = np.argsort(riser_t)
    riser_t = riser_t[order]
    riser_top = riser_top[order]
    if len(riser_t) > 0:
        base_t = float((qpos[0:2] - climb_origin) @ climb_axis)
        nr = len(riser_t)
        # body x 轴在爬升方向上的投影：训练 env 中 yaw=pi/2 时 sin(yaw)=1；
        # 通用形式为 cos(yaw - target_heading)
        sy = float(np.cos(yaw - target_heading))
        for k, off in enumerate((FRONT_REAR_OFFSET, -FRONT_REAR_OFFSET)):
            at = base_t + off * sy
            idx_p = int(np.sum(riser_t < at))
            nxt = min(idx_p, nr - 1)
            d_next = riser_t[nxt] - at
            top_next = riser_top[nxt]
            cur_top = riser_top[max(idx_p - 1, 0)] if idx_p > 0 else 0.0
            ctx[k] = float(np.clip(d_next, -1.0, 1.0))
            ctx[k + 2] = float(np.clip((top_next - cur_top) / 0.2, -1.0, 1.0))
    rough = np.array([1.0 if len(riser_t) > 0 else 0.0], dtype=np.float32)
    return np.concatenate([angvel, g, np.asarray(cmd, dtype=np.float32), leg_err,
                           leg_vel, np.asarray(last_action, dtype=np.float32),
                           heading, ctx, rough]).astype(np.float32)
