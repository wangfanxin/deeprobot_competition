"""S10 MJX RL env (functional, mujoco_playground style).

Single-env functional step with top-level jax.vmap batching, all
obs/reward/done computed in JAX on GPU. 50 Hz policy, 200 Hz sim (decimation 4).
"""
import os, hashlib
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from rl_stair.envs.terrain import Terrain, build_model_xml, stairs, ridge, flat, ASSET_DIR


@dataclass
class S10RLCfg:
    num_envs: int = 2048
    sim_dt: float = 0.005
    decimation: int = 4
    seed: int = 0
    njmax: int = 400
    naconmax: int = 20

    # PD
    # kp=80 tried 05:25 (S10 stair WBC value) but made T2c(8cm roll-climb) regress
    # repeatedly vs kp=50 (0.35-0.5). Reverted 05:58. Reward levers (r_riser=3, terrain
    # height) retained for the >8cm lift-climb.
    kp_leg: float = 50.0
    kd_leg: float = 1.0
    kp_wheel: float = 2.0
    # S10-adapted (NOT go2w): FK-verified wheel lift within +/-action_scale
    #   0.25->4.2cm 0.5->9.9cm 0.6->12.4cm 0.7->14.7cm 0.8->16.7cm
    #   competition riser = 12.5cm -> action_scale=0.7 (go2w 0.25 insufficient; 0.8
    #   tried 00:35 but destabilized lower stages (overshoot) and was reverted 01:05)
    action_scale: float = 0.7
    # S10-adapted: wheel r=0.081; go2w vel_scale=10 caps at 0.81 m/s but stair entry
    #   is ~1.5 m/s -> vel_scale=24 gives 1.94 m/s max (cmd 0.8-1.8 covered)
    vel_scale: float = 24.0
    torque_clip_leg: float = 48.0
    torque_clip_wheel: float = 13.5

    # handoff sampler
    vx_lo, vx_hi = -0.5, 2.5
    vy_lo, vy_hi = -0.3, 0.3
    yaw_lo, yaw_hi = -0.5, 0.5
    h_off = 0.05
    q_off = 0.1
    v_off = 1.0
    spawn_x = 0.3
    # USER-DIRECTED 2026-08-16: initial-pose domain randomization (handoff robustness).
    # squat_frac = fraction of resets starting at the cruise SQUAT pose (lower body,
    # legs -1.10/1.90; verified squat-start = 0/30 on the real mesh without training).
    # leg_q_jit = joint-angle jitter (rad) on the legs for ALL resets (real-mesh tol:
    # jit 0.15->93%, 0.3->70%, 0.5->30%; T6 target >= 0.3). Off by default (0.0) so
    # early stages are unchanged; T6_handoff enables them.
    squat_frac: float = 0.0
    leg_q_jit: float = 0.0
    # spawn distance before first riser (for "close-spawn lift practice" stages 06:50)
    spawn_back_lo: float = 0.5
    spawn_back_hi: float = 2.0

    cmd_vx_lo, cmd_vx_hi = 0.8, 1.8

    max_ep_len: int = 500
    fall_z: float = 0.15
    reset_backtrack: float = 0.5
    tilt_limit: float = 1.2

    r_progress = 4.0   # +2->+4 (2026-08-14): 强化前进激励，破除"站立吃小奖励"局部最优；legged_gym 主用 velocity-tracking，progress 为辅助，翻倍不改变最优方向
    r_riser = 3.0   # 1->3 (2026-08-15 04:00): 4x10cm排台阶卡3次 - 单级可靠率~50%需74%; riser奖励1.0相对tracking太弱, 强化跨台阶信号
    r_goal = 10.0
    r_termination = -0.8
    r_speed = 2.0
    # Domain randomization (Phase 1, 95% plan): per-episode PD/torque scales + push.
    dr_kp_leg_lo, dr_kp_leg_hi = 0.8, 1.2
    dr_kd_leg_lo, dr_kd_leg_hi = 0.8, 1.2
    # 2026-08-15 22:45 (sim2sim): MJX<->C++ wheel drive measured 2.7x gap (§3.36);
    # widen WHEEL DR to cover it so the policy stays fast/robust with weak wheels.
    # 2026-08-16 (real-mesh upright climb): widen wheel-drive DR to cover the documented
    # 2.7x MJX<->real-mesh wheel gap (real wheels ~0.37x). The integrated real-mesh climb
    # stalls at the 0.125m riser (vx -0.4..-0.8 backward slide) because the policy relies
    # on momentum it cannot build with weak wheels; training with 0.3-1.2x forces the
    # policy to learn lift-over technique that works with weak wheels.
    dr_kp_wheel_lo, dr_kp_wheel_hi = 0.3, 1.2
    dr_tclip_lo, dr_tclip_hi = 0.8, 1.2
    dr_tclip_wheel_lo, dr_tclip_wheel_hi = 0.8, 1.2
    push_interval_steps: int = 750
    push_vel: float = 1.0
    # ---- 4-item stair shaping (2026-08-15, doc RL_stair_奖励增强_4项_20260815.md, APPROVED plan b) ----
    # 1.1 base-contact termination (geometric; Isaac Lab base_contact->done; stand clearance ~0.28m)
    enable_base_contact: bool = True
    base_contact_margin: float = 0.04
    # 1.2 knee/leg scrape penalty (geometric proxy of Isaac Lab undesired_contacts .*THIGH -1.0)
    enable_scrape: bool = True
    r_scrape: float = -1.0
    scrape_margin: float = 0.02
    # 1.3 front-wheel edge clearance shaping (target-height gaussian, NOT 'higher is better')
    enable_wheel_clear: bool = True
    r_wheel_clear: float = 0.5
    wheel_clear_window: float = 0.15
    wheel_clear_sigma: float = 0.05
    # motion gate: only reward clearance while advancing (checklist: "low-speed/standing
    # must not reward idle lift"; prevents hovering-next-to-step local optimum at small
    # risers where R(8.1cm) > riser(5cm) -> normal wheel height already above top)
    wheel_clear_min_vx: float = 0.3
    # 1.4 wheel-stuck-at-riser-face penalty (narrow window; must not punish legit roll-up climb)
    enable_wheel_stumble: bool = True
    r_wheel_stumble: float = -1.0
    wheel_stumble_margin: float = 0.02
    wheel_stumble_window: float = 0.10
    # critic real contact forces via MJX efc_force (critic-only, deploy unaffected)
    use_real_cfrc: bool = True
    r_orientation = -2.0
    r_height = -0.5
    r_torque = -0.0001
    r_action_rate = -0.0002
    r_dof_limits = -0.9
    r_hip_l2 = -0.1
    # clip dense rewards at 0, termination added AFTER clip
    # (go2w_rl_gym/legged_gym `only_positive_rewards = True`; avoids negative-return std inflation)
    only_positive_rewards: bool = True
    # velocity tracking (legged_gym/go2w_rl_gym primary locomotion reward)
    r_tracking_lin_vel = 4.0
    r_tracking_ang_vel = 2.0
    tracking_sigma = 0.25
    # Chamorro et al. (ICRA24 2402.06143): actor obs has goal-direction(2)+heading-error(1),
    # reward keeps yaw aligned to task axis. Our task axis = world +y (track), target yaw = pi/2.
    r_heading = 2.0
    # r_speed (07:10): policy approaches steps at ~0.5m/s (cmd 0.8-1.8 untracked) -> no
    # momentum to carry wheels over >radius steps. Dense forward-speed reward (user: improve
    # climb time; momentum helps above-radius lift-climb). progress(4/m) too weak (0.04/step).

    terrain: Terrain = field(default_factory=flat)


class S10RLEnv:
    def __init__(self, cfg: S10RLCfg):
        self.cfg = cfg
        self.n = cfg.num_envs
        self.dt = cfg.sim_dt

        os.makedirs(ASSET_DIR, exist_ok=True)
        xml = build_model_xml(cfg.terrain)
        h = hashlib.md5(xml.encode()).hexdigest()[:10]
        self.xml_path = os.path.join(ASSET_DIR, f"s10_train_{h}.xml")
        if not os.path.exists(self.xml_path):
            with open(self.xml_path, "w", encoding="utf-8") as f:
                f.write(xml)
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model = mjx.put_model(self.mj_model)

        self.qpos0 = np.array(self.mj_model.qpos0, dtype=np.float64)
        self.nq, self.nv, self.nu = self.mj_model.nq, self.mj_model.nv, self.mj_model.nu
        self.jnt_adr = np.array(self.mj_model.jnt_qposadr)
        names = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(self.mj_model.njnt)]
        self.leg_idx = jnp.asarray([i for i in range(16) if "wheel" not in names[i + 1]])
        self.wheel_idx = jnp.asarray([i for i in range(16) if "wheel" in names[i + 1]])
        self.hipx_idx = jnp.asarray([i for i in range(16) if "hipx" in names[i + 1]])
        self.act2jnt = jnp.asarray([self.jnt_adr[i + 1] for i in range(16)])   # qpos adr
        self.act2vel = jnp.asarray([self.jnt_adr[i + 1] - 1 for i in range(16)])  # qvel adr
        self.jnt_range = jnp.asarray(self.mj_model.jnt_range)[jnp.asarray([i + 1 for i in range(16)])]

        # terrain metadata as jax arrays
        self.riser_y = jnp.asarray(cfg.terrain.riser_y, dtype=jnp.float32)
        self.riser_top = jnp.asarray(cfg.terrain.riser_top_z, dtype=jnp.float32)
        self.start_y = float(cfg.terrain.start_y)
        self.goal_y = float(cfg.terrain.goal_y)
        self.first_riser_y = float(cfg.terrain.riser_y[0]) if cfg.terrain.riser_y else None

        # exact terrain height from box list (covers stairs/ridge/mixed; flat -> 0)
        _bx = cfg.terrain.boxes
        if len(_bx) > 0:
            self.box_y_lo = jnp.asarray([b["pos"][1] - b["size"][1] for b in _bx], dtype=jnp.float32)
            self.box_y_hi = jnp.asarray([b["pos"][1] + b["size"][1] for b in _bx], dtype=jnp.float32)
            self.box_x_lo = jnp.asarray([b["pos"][0] - b["size"][0] for b in _bx], dtype=jnp.float32)
            self.box_x_hi = jnp.asarray([b["pos"][0] + b["size"][0] for b in _bx], dtype=jnp.float32)
            self.box_top = jnp.asarray([b["pos"][2] + b["size"][2] for b in _bx], dtype=jnp.float32)
        else:
            self.box_y_lo = self.box_y_hi = self.box_x_lo = self.box_x_hi = self.box_top = jnp.zeros(0, dtype=jnp.float32)

        # body ids for knees/wheels (contact force grouping + scrape/clearance geometry)
        _mjb = mujoco.mjtObj.mjOBJ_BODY
        self.knee_body_ids = jnp.asarray([mujoco.mj_name2id(self.mj_model, _mjb, n)
                                          for n in ("fl_knee","fr_knee","hl_knee","hr_knee")], dtype=jnp.int32)
        self.wheel_body_ids = jnp.asarray([mujoco.mj_name2id(self.mj_model, _mjb, n)
                                           for n in ("fl_wheel","fr_wheel","hl_wheel","hr_wheel")], dtype=jnp.int32)
        self.geom_bodyid = jnp.asarray(self.mj_model.geom_bodyid, dtype=jnp.int32)

        # standing pose = S10 stair-appropriate TALL stance (USER-DIRECTED 2026-08-14:
        # do NOT copy the cruise half-squat - that pose presses the body low for speed;
        # stair climbing needs a HIGHER body with LESS-bent legs).
        #   fl:[-0.05,-0.60, 1.20] fr:[ 0.05,-0.60, 1.20]
        #   hl:[-0.05, 0.60,-1.20] hr:[ 0.05, 0.60,-1.20]   (wheel=0)
        # MuJoCo verified: settled stand_z=0.359 (vs cruise 0.262), wheel lift @act0.7 =
        # 0.193m (vs cruise 0.147), half-wheelbase 0.246. PD-hold stable.
        self.default_dof = jnp.asarray(np.array(
            [-0.05, -0.60, 1.20, 0.0,
              0.05, -0.60, 1.20, 0.0,
             -0.05,  0.60, -1.20, 0.0,
              0.05,  0.60, -1.20, 0.0], dtype=np.float64))
        self.stand_qpos = jnp.asarray(self._compute_stand())
        self.stand_z = float(np.asarray(self.stand_qpos)[2])

        self.obs_dim = 3 + 3 + 2 + 12 + 12 + 16 + 2 + 4 + 1   # +2 heading(Chamorro)
        self.priv_dim = self.obs_dim + 3 + 1 + 4 * 3 + 1
        self.action_size = 16
        # batched empty data (for reset.replace)
        self._empty_data = jax.vmap(
            lambda _: mjx.make_data(self.model, njmax=cfg.njmax, naconmax=cfg.naconmax)
        )(jnp.zeros(cfg.num_envs))

    def _compute_stand(self):
        # settle with PD holding default_dof (go2w pose) from slightly high -> equilibrium
        m, d = self.mj_model, mujoco.MjData(self.mj_model)
        default = np.asarray(self.default_dof)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
        leg_idx = [i for i, nm in enumerate(names) if "wheel" not in nm]
        wheel_idx = [i for i, nm in enumerate(names) if "wheel" in nm]
        adr = np.array(m.jnt_qposadr)[1:]
        d.qpos[:] = self.qpos0
        for i in range(16):
            d.qpos[adr[i]] = default[i]
        d.qpos[2] = 0.42
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(1500):
            q = d.qpos[adr] - default
            qd = d.qvel[adr - 1]
            tau = np.zeros(16)
            lt = np.clip(50.0 * (0.0 - q) - 1.0 * qd, -48.0, 48.0)
            tau[leg_idx] = lt[leg_idx]
            wt = np.clip(2.0 * (0.0 - qd), -13.5, 13.5)
            tau[wheel_idx] = wt[wheel_idx]
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        return np.array(d.qpos)

    # -------------------------------------------------------------- core
    def _pd(self, data, action, dr=None):
        q = data.qpos[:, self.act2jnt] - self.default_dof
        qd = data.qvel[:, self.act2vel]
        tau = jnp.zeros((self.n, 16))
        q_target = action * self.cfg.action_scale
        kp_l = (self.cfg.kp_leg * dr["kp_leg"][:, None]) if dr is not None else self.cfg.kp_leg
        kd_l = (self.cfg.kd_leg * dr["kd_leg"][:, None]) if dr is not None else self.cfg.kd_leg
        tl = (self.cfg.torque_clip_leg * dr["tclip_leg"][:, None]) if dr is not None else self.cfg.torque_clip_leg
        leg_tau = kp_l * (q_target - q) - kd_l * qd
        leg_tau = jnp.clip(leg_tau, -tl, tl)
        tau = tau.at[:, self.leg_idx].set(leg_tau[:, self.leg_idx])
        vel_ref = action * self.cfg.vel_scale
        kp_w = (self.cfg.kp_wheel * dr["kp_wheel"][:, None]) if dr is not None else self.cfg.kp_wheel
        tw = (self.cfg.torque_clip_wheel * dr["tclip_wheel"][:, None]) if dr is not None else self.cfg.torque_clip_wheel
        w_tau = kp_w * (vel_ref - qd)
        w_tau = jnp.clip(w_tau, -tw, tw)
        tau = tau.at[:, self.wheel_idx].set(w_tau[:, self.wheel_idx])
        return tau

    def _terrain_h(self, ys):
        """Exact terrain top height at world-y positions (boxes span full track width)."""
        if self.box_top.shape[0] == 0:
            return jnp.zeros_like(ys, dtype=jnp.float32)
        inside = (self.box_y_lo[None, :] <= ys[..., None]) & (ys[..., None] <= self.box_y_hi[None, :])
        return jnp.max(jnp.where(inside, self.box_top[None, :], 0.0), axis=-1)

    def _wheel_contact_forces(self, data):
        """Per-wheel contact force (n,12) from MJX efc_force + contact frame (critic-only).
        Handles batched (n,ncon) and unbatched (ncon,) arrays (pre-step reset)."""
        n = data.qpos.shape[0]
        zeros = jnp.zeros((n, 12), dtype=jnp.float32)
        c = data.contact
        g1, g2 = c.geom1, c.geom2
        if g1.ndim == 1:      # pre-step data: no forces yet
            return zeros
        nefc = data.efc_force.shape[-1]
        addr = jnp.clip(c.efc_address, 0, nefc - 1)          # (ncon,) shared layout
        valid = (c.efc_address >= 0)[None, :, None]          # (1,ncon,1)
        F = jnp.where(valid, data.efc_force[:, addr][..., None], 0.0) * c.frame[..., 0, :]  # normal = first row
        out = []
        for b in self.wheel_body_ids:
            mask = (self.geom_bodyid[g1] == b) | (self.geom_bodyid[g2] == b)
            out.append(jnp.sum(jnp.where(mask[..., None], F, 0.0), axis=1))
        return jnp.concatenate(out, axis=-1)

    def _terrain_ctx(self, data):
        base_y = data.qpos[:, 1]
        n = base_y.shape[0]
        ctx = jnp.zeros((n, 4), dtype=jnp.float32)
        ry = self.riser_y
        rz = self.riser_top
        if len(ry) == 0:
            return ctx
        nr = ry.shape[0]
        # yaw-rotate front/rear axle offsets (approach-angle randomization 09:55): wheels at
        # body-x +-0.246 -> world-y offset = 0.246*sin(yaw). At yaw=pi/2 -> +-0.246.
        _qw, _qx, _qy, _qz = data.qpos[:, 3], data.qpos[:, 4], data.qpos[:, 5], data.qpos[:, 6]
        _yaw = jnp.arctan2(2*(_qw*_qz + _qx*_qy), 1 - 2*(_qy*_qy + _qz*_qz))
        _s = jnp.sin(_yaw)
        for k, off in enumerate((0.246, -0.246)):   # tall-stance half-wheelbase
            ay = base_y + off * _s
            idx = jnp.sum(ry[None, :] < ay[:, None], axis=-1)   # count risers passed
            nxt = jnp.minimum(idx, nr - 1)
            d_next = ry[nxt] - ay
            top_next = rz[nxt]
            cur_top = jnp.where(idx > 0, rz[jnp.maximum(idx - 1, 0)], 0.0)
            ctx = ctx.at[:, k].set(jnp.clip(d_next, -1.0, 1.0))
            ctx = ctx.at[:, k + 2].set(jnp.clip((top_next - cur_top) / 0.2, -1.0, 1.0))
        return ctx

    def _obs(self, data, last_action, cmd):
        qpos = data.qpos
        q = qpos[:, self.act2jnt]
        qd = data.qvel[:, self.act2vel]
        qw, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
        g = jnp.stack([2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx),
                       1 - 2*(qx*qx + qy*qy)], axis=-1)
        angvel = data.qvel[:, 3:6] * 0.25
        leg_err = q[:, self.leg_idx] - self.default_dof[self.leg_idx]
        leg_vel = qd[:, self.leg_idx] * 0.05
        rough = jnp.full((qpos.shape[0], 1), 1.0 if len(self.riser_y) > 0 else 0.0)
        # Chamorro-style heading feedback: yaw error vs task axis (+y, target pi/2),
        # encoded as [cos,sin] (wrap-safe, bounded). Deployment uses track heading.
        yaw = jnp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        yaw_err = yaw - jnp.pi/2.0
        heading = jnp.stack([jnp.cos(yaw_err), jnp.sin(yaw_err)], axis=-1)
        obs = jnp.concatenate([angvel, g, cmd,
                               leg_err, leg_vel, last_action,
                               heading, self._terrain_ctx(data), rough], axis=-1)
        return obs

    def _priv(self, data, obs):
        if self.cfg.use_real_cfrc:
            # Chamorro ICRA24: contact force x0.01 in critic; go2w normalizes or omits;
            # legged_gym clip_observations=100. Raw N (~240) would badly condition the
            # critic -> scale 0.01 + clip +-5 (raw +-500N).
            cfrc = jnp.clip(self._wheel_contact_forces(data) * 0.01, -5.0, 5.0)
        else:
            cfrc = jnp.zeros((obs.shape[0], 12), dtype=jnp.float32)
        priv = jnp.concatenate([obs, data.qvel[:, 0:3], data.qpos[:, 2:3], cfrc,
                                jnp.full((obs.shape[0], 1), 0.8)], axis=-1)
        return priv

    def _reward(self, data, state, tau, action, new_progress, new_risers, success, term, timeout):
        qpos = data.qpos
        qw, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
        roll = jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1, 1))
        q = qpos[:, self.act2jnt]
        r = jnp.zeros(qpos.shape[0])
        r += self.cfg.r_progress * jnp.maximum(0.0, new_progress - state["prev_progress"])
        r += self.cfg.r_riser * jnp.clip(new_risers - state["riser_crossed"], 0, None)
        r += self.cfg.r_goal * success
        r += self.cfg.r_orientation * (roll**2 + pitch**2)
        # height target = stand_z + terrain height under base (stairs-aware; old flat-ground
        # version PENALIZED being on top of a step = anti-climb. 2026-08-15 05:10)
        if len(self.riser_y) > 0:
            _idx = jnp.sum(self.riser_y[None, :] < qpos[:, 1:2], axis=-1).astype(jnp.int32)
            _th = jnp.where(_idx > 0, self.riser_top[jnp.maximum(_idx - 1, 0)], 0.0)
        else:
            _th = jnp.zeros_like(qpos[:, 1])
        r += self.cfg.r_height * (qpos[:, 2] - (self.stand_z + _th))**2
        r += self.cfg.r_torque * jnp.sum(tau**2, axis=-1)
        r += self.cfg.r_action_rate * jnp.sum((action - state["last_action"])**2, axis=-1)
        soft = 0.9
        over = jnp.maximum(q - soft*self.jnt_range[:, 1], 0.0)**2 + \
               jnp.maximum(soft*self.jnt_range[:, 0] - q, 0.0)**2
        r += self.cfg.r_dof_limits * jnp.sum(over[:, self.leg_idx], axis=-1)
        r += self.cfg.r_hip_l2 * jnp.sum(action[:, self.hipx_idx]**2, axis=-1)
        # velocity tracking: go2w/legged_gym body-frame lin vel, exp(-err/sigma)
        qw, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
        vl = data.qvel[:, 0:3]
        tx = 2.0 * (-qy * vl[:, 2] + qz * vl[:, 1])
        ty = 2.0 * (-qz * vl[:, 0] + qx * vl[:, 2])
        tz = 2.0 * (-qx * vl[:, 1] + qy * vl[:, 0])
        vbx = vl[:, 0] + qw * tx + (-qy * tz + qz * ty)
        vby = vl[:, 1] + qw * ty + (-qz * tx + qx * tz)
        lin_err = (state["cmd"][:, 0] - vbx) ** 2 + (0.0 - vby) ** 2
        r += self.cfg.r_tracking_lin_vel * jnp.exp(-lin_err / self.cfg.tracking_sigma)
        ang_err = (state["cmd"][:, 1] - data.qvel[:, 5]) ** 2
        r += self.cfg.r_tracking_ang_vel * jnp.exp(-ang_err / self.cfg.tracking_sigma)
        # dense forward-speed reward (momentum for >radius step clearing)
        r += self.cfg.r_speed * jnp.maximum(0.0, vbx)
        # Chamorro-style heading alignment: keep yaw at task axis (+y). Directly
        # penalizes the measured "circling" (wz~0.4 rad/s, y_max capped ~3.8m).
        yaw = jnp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        yaw_err = yaw - jnp.pi/2.0
        r += self.cfg.r_heading * jnp.exp(-(yaw_err**2) / self.cfg.tracking_sigma)
        # ---- 4-item stair shaping (doc RL_stair_奖励增强_4项_20260815.md) ----
        if self.cfg.enable_scrape:
            kx = data.xpos[:, self.knee_body_ids, 0]
            ky = data.xpos[:, self.knee_body_ids, 1]
            kz = data.xpos[:, self.knee_body_ids, 2]
            th_k = self._terrain_h(ky)
            scrape = jnp.sum(jnp.clip(th_k + self.cfg.scrape_margin - kz, 0, None), axis=-1)
            r += self.cfg.r_scrape * scrape
        nr = self.riser_y.shape[0]
        if nr > 0:
            wy = data.xpos[:, self.wheel_body_ids, 1]
            wz = data.xpos[:, self.wheel_body_ids, 2]
            idx = jnp.sum(self.riser_y[None, :] < wy[:, :, None], axis=-1)
            nxt = jnp.minimum(idx, nr - 1)
            ry_next = self.riser_y[nxt]
            top_next = self.riser_top[nxt]
            d = wy - ry_next
            if self.cfg.enable_wheel_clear:
                # body-frame forward speed (same transform as tracking reward below)
                _qw,_qx,_qy,_qz = data.qpos[:,3], data.qpos[:,4], data.qpos[:,5], data.qpos[:,6]
                _vl = data.qvel[:, 0:3]
                _tx = 2.0*(-_qy*_vl[:,2] + _qz*_vl[:,1]); _ty = 2.0*(-_qz*_vl[:,0] + _qx*_vl[:,2])
                _tz = 2.0*(-_qx*_vl[:,1] + _qy*_vl[:,0])
                _vbx = _vl[:,0] + _qw*_tx + (-_qy*_tz + _qz*_ty)
                moving = _vbx > self.cfg.wheel_clear_min_vx
                in_win = jnp.abs(d) < self.cfg.wheel_clear_window
                clear = jnp.exp(-((wz - top_next) / self.cfg.wheel_clear_sigma) ** 2)
                r += self.cfg.r_wheel_clear * jnp.sum(jnp.where(in_win & moving[:, None], clear, 0.0), axis=-1)
            if self.cfg.enable_wheel_stumble:
                stuck = (d > 0) & (d < self.cfg.wheel_stumble_window) & (wz < top_next - self.cfg.wheel_stumble_margin)
                r += self.cfg.r_wheel_stumble * jnp.sum(
                    jnp.where(stuck, jnp.clip(top_next - wz, 0, None), 0.0), axis=-1)
        if self.cfg.only_positive_rewards:
            r = jnp.clip(r, 0.0, None)
        r = r + self.cfg.r_termination * term
        return r

    def reset(self, rng):
        n = self.n
        rng, k1 = jax.random.split(rng)
        rng, k2 = jax.random.split(rng)
        # face the course (+y): body +x forward must equal world +y (progress axis),
        # matching the competition track direction; yaw_lo/hi are perturbation around pi/2
        yaw = np.pi / 2 + jax.random.uniform(k1, (n,), minval=self.cfg.yaw_lo, maxval=self.cfg.yaw_hi)
        vx = jax.random.uniform(k1, (n,), minval=self.cfg.vx_lo, maxval=self.cfg.vx_hi)
        vy = jax.random.uniform(k2, (n,), minval=self.cfg.vy_lo, maxval=self.cfg.vy_hi)
        vyaw = jax.random.uniform(k2, (n,), minval=-0.5, maxval=0.5)
        h_off = jax.random.uniform(k1, (n,), minval=-self.cfg.h_off, maxval=self.cfg.h_off)
        x_spawn = jax.random.uniform(k2, (n,), minval=-self.cfg.spawn_x, maxval=self.cfg.spawn_x)
        q_off = jax.random.uniform(k1, (n, 16), minval=-self.cfg.q_off, maxval=self.cfg.q_off)
        # USER-DIRECTED 2026-08-16: initial-pose DR - squat_frac resets start at the
        # cruise squat (lower body), rest at tall default_dof; all get leg_q_jit on the
        # legs so the policy learns to stand up + climb from the handoff leg state.
        _sq16 = jnp.asarray([-0.05,-1.10,1.90,0.0, 0.05,-1.10,1.90,0.0,
                             -0.05,1.10,-1.90,0.0, 0.05,1.10,-1.90,0.0], dtype=jnp.float32)
        _is_sq = jax.random.uniform(jax.random.fold_in(k1, 999), (n,)) < self.cfg.squat_frac
        _base = jnp.where(_is_sq[:, None], _sq16[None, :], self.default_dof[None, :])
        if self.cfg.leg_q_jit > 0.0:
            _qj = jax.random.uniform(k1, (n, 16), minval=-self.cfg.leg_q_jit,
                                     maxval=self.cfg.leg_q_jit)
            q_off = q_off.at[:, self.leg_idx].set(_qj[:, self.leg_idx])
        jv = jax.random.uniform(k2, (n, 16), minval=-self.cfg.v_off, maxval=self.cfg.v_off)

        if self.first_riser_y is not None:
            sy = self.first_riser_y - jax.random.uniform(k1, (n,), minval=self.cfg.spawn_back_lo, maxval=self.cfg.spawn_back_hi)
        else:
            sy = jax.random.uniform(k1, (n,), minval=-2.0, maxval=0.0)

        qpos = jnp.broadcast_to(self.stand_qpos, (n, self.nq))
        c, s = jnp.cos(yaw*0.5), jnp.sin(yaw*0.5)
        cy, sy = jnp.cos(yaw), jnp.sin(yaw)   # BUGFIX: 速度旋转用全角 yaw（quat 才用半角）
        qpos = qpos.at[:, 3:7].set(jnp.stack([c, jnp.zeros_like(c), jnp.zeros_like(c), s], axis=-1))
        qpos = qpos.at[:, 0].set(x_spawn)
        qpos = qpos.at[:, 1].set(sy)
        qpos = qpos.at[:, 2].set(self.stand_z + h_off)
        qpos = qpos.at[:, self.act2jnt].set(_base + q_off)
        qvel = jnp.zeros((n, self.nv))
        qvel = qvel.at[:, 0].set(cy*vx - sy*vy)
        qvel = qvel.at[:, 1].set(sy*vx + cy*vy)
        qvel = qvel.at[:, 5].set(vyaw)
        qvel = qvel.at[:, self.act2vel].add(jv)

        cmd = jnp.stack([jax.random.uniform(rng, (n,), minval=self.cfg.cmd_vx_lo,
                                            maxval=self.cfg.cmd_vx_hi),
                         jnp.zeros(n)], axis=-1)
        data = self._empty_data.replace(qpos=jax.lax.stop_gradient(qpos),
                                        qvel=jax.lax.stop_gradient(qvel))
        rng, k3 = jax.random.split(rng)
        dr = {
            "kp_leg": jax.random.uniform(jax.random.fold_in(k3, 0), (n,), minval=self.cfg.dr_kp_leg_lo, maxval=self.cfg.dr_kp_leg_hi),
            "kd_leg": jax.random.uniform(jax.random.fold_in(k3, 1), (n,), minval=self.cfg.dr_kd_leg_lo, maxval=self.cfg.dr_kd_leg_hi),
            "kp_wheel": jax.random.uniform(jax.random.fold_in(k3, 2), (n,), minval=self.cfg.dr_kp_wheel_lo, maxval=self.cfg.dr_kp_wheel_hi),
            "tclip_leg": jax.random.uniform(jax.random.fold_in(k3, 3), (n,), minval=self.cfg.dr_tclip_lo, maxval=self.cfg.dr_tclip_hi),
            "tclip_wheel": jax.random.uniform(jax.random.fold_in(k3, 4), (n,), minval=self.cfg.dr_tclip_wheel_lo, maxval=self.cfg.dr_tclip_wheel_hi),
            "push_angle": jax.random.uniform(jax.random.fold_in(k3, 5), (n,), minval=0.0, maxval=2.0*np.pi),
        }
        state = {
            "data": data,
            "rng": rng,
            "dr": dr,
            "ep_len": jnp.zeros(n, dtype=jnp.int32),
            "last_action": jnp.zeros((n, 16)),
            "cmd": cmd,
            "prev_progress": jnp.maximum(0.0, sy - self.start_y),   # init to spawn progress (no free bonus, legged_gym convention)
            "riser_crossed": jnp.zeros(n, dtype=jnp.int32),
            "done": jnp.zeros(n, dtype=bool),
            "success": jnp.zeros(n, dtype=bool),
        }
        obs = self._obs(data, state["last_action"], cmd)
        priv = self._priv(data, obs)
        return state, obs, priv

    def step(self, state, action):
        a = jnp.clip(action, -1.0, 1.0)
        tau = self._pd(state["data"], a, state.get("dr"))
        data = state["data"]
        def _one(dd, _):
            return jax.vmap(lambda x: mjx.step(self.model, x))(dd.replace(ctrl=tau)), None
        data = jax.lax.scan(_one, data, None, self.cfg.decimation)[0]
        # DR push disturbance (legged_gym _push_robots)
        if self.cfg.push_vel > 0.0:
            _pm = (state["ep_len"] % self.cfg.push_interval_steps) == 0
            _pa = state["dr"]["push_angle"] if "dr" in state else jnp.zeros(self.n)
            _dir = jnp.stack([jnp.cos(_pa), jnp.sin(_pa)], axis=-1) * self.cfg.push_vel
            data = data.replace(qvel=data.qvel.at[:, 0:2].add(_pm[:, None].astype(jnp.float32) * _dir))

        base_y = data.qpos[:, 1]
        base_z = data.qpos[:, 2]
        new_progress = jnp.maximum(0.0, base_y - self.start_y)
        if len(self.riser_y) > 0:
            new_risers = jnp.sum(self.riser_y[None, :] < base_y[:, None], axis=-1).astype(jnp.int32)
        else:
            new_risers = jnp.zeros_like(base_y, dtype=jnp.int32)

        qw, qx, qy, qz = data.qpos[:, 3], data.qpos[:, 4], data.qpos[:, 5], data.qpos[:, 6]
        roll = jnp.abs(jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
        pitch = jnp.abs(jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1, 1)))
        success = (base_y > self.goal_y) & (base_z > self.cfg.fall_z)
        fall = base_z < self.cfg.fall_z
        backtrack = base_y < self.start_y - self.cfg.reset_backtrack
        term = fall | backtrack | (roll > self.cfg.tilt_limit) | (pitch > self.cfg.tilt_limit)
        if self.cfg.enable_base_contact:
            body_contact = base_z < self._terrain_h(base_y) + self.cfg.base_contact_margin
            term = term | body_contact
        ep_len = state["ep_len"] + 1
        timeout = ep_len >= self.cfg.max_ep_len
        done = term | timeout | success

        reward = self._reward(data, state, tau, a, new_progress, new_risers, success, term, timeout)
        new_state = {
            "data": data,
            "rng": state["rng"],
            "dr": state.get("dr"),
            "ep_len": ep_len,
            "last_action": a,
            "cmd": state["cmd"],
            "prev_progress": new_progress,
            "riser_crossed": new_risers,
            "done": done,
            "success": success,
        }
        obs = self._obs(data, a, state["cmd"])
        priv = self._priv(data, obs)
        return new_state, obs, priv, reward, done, success

    # ------------------------------------------------------------ utilities
    def obs_of(self, state):
        return self._obs(state["data"], state["last_action"], state["cmd"])

    def priv_of(self, state, obs):
        return self._priv(state["data"], obs)

    def merge_reset(self, state, done, reset_state):
        """Apply reset_state to envs where done is True (functional)."""
        def _w(x, y):
            d = done if x.ndim <= 1 else done.reshape([-1] + [1] * (x.ndim - 1))
            return jnp.where(d, x, y)
        out = {}
        for k in state.keys():
            if k == "rng":
                out[k] = state[k]
                continue
            out[k] = jax.tree.map(_w, reset_state[k], state[k])
        return out

    def reset_state(self, rng):
        """Full reset returning only the state dict (for merge_reset)."""
        rng, k1 = jax.random.split(rng)
        rng, k2 = jax.random.split(rng)
        n = self.n
        # face the course (+y): body +x forward must equal world +y (progress axis),
        # matching the competition track direction; yaw_lo/hi are perturbation around pi/2
        yaw = np.pi / 2 + jax.random.uniform(k1, (n,), minval=self.cfg.yaw_lo, maxval=self.cfg.yaw_hi)
        vx = jax.random.uniform(k1, (n,), minval=self.cfg.vx_lo, maxval=self.cfg.vx_hi)
        vy = jax.random.uniform(k2, (n,), minval=self.cfg.vy_lo, maxval=self.cfg.vy_hi)
        vyaw = jax.random.uniform(k2, (n,), minval=-0.5, maxval=0.5)
        h_off = jax.random.uniform(k1, (n,), minval=-self.cfg.h_off, maxval=self.cfg.h_off)
        x_spawn = jax.random.uniform(k2, (n,), minval=-self.cfg.spawn_x, maxval=self.cfg.spawn_x)
        q_off = jax.random.uniform(k1, (n, 16), minval=-self.cfg.q_off, maxval=self.cfg.q_off)
        # USER-DIRECTED 2026-08-16: initial-pose DR - squat_frac resets start at the
        # cruise squat (lower body), rest at tall default_dof; all get leg_q_jit on the
        # legs so the policy learns to stand up + climb from the handoff leg state.
        _sq16 = jnp.asarray([-0.05,-1.10,1.90,0.0, 0.05,-1.10,1.90,0.0,
                             -0.05,1.10,-1.90,0.0, 0.05,1.10,-1.90,0.0], dtype=jnp.float32)
        _is_sq = jax.random.uniform(jax.random.fold_in(k1, 999), (n,)) < self.cfg.squat_frac
        _base = jnp.where(_is_sq[:, None], _sq16[None, :], self.default_dof[None, :])
        if self.cfg.leg_q_jit > 0.0:
            _qj = jax.random.uniform(k1, (n, 16), minval=-self.cfg.leg_q_jit,
                                     maxval=self.cfg.leg_q_jit)
            q_off = q_off.at[:, self.leg_idx].set(_qj[:, self.leg_idx])
        jv = jax.random.uniform(k2, (n, 16), minval=-self.cfg.v_off, maxval=self.cfg.v_off)
        if self.first_riser_y is not None:
            sy = self.first_riser_y - jax.random.uniform(k1, (n,), minval=self.cfg.spawn_back_lo, maxval=self.cfg.spawn_back_hi)
        else:
            sy = jax.random.uniform(k1, (n,), minval=-2.0, maxval=0.0)
        qpos = jnp.broadcast_to(self.stand_qpos, (n, self.nq))
        c, s = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
        cy, sy = jnp.cos(yaw), jnp.sin(yaw)   # BUGFIX: 速度旋转用全角 yaw
        qpos = qpos.at[:, 3:7].set(jnp.stack([c, jnp.zeros_like(c), jnp.zeros_like(c), s], axis=-1))
        qpos = qpos.at[:, 0].set(x_spawn)
        qpos = qpos.at[:, 1].set(sy)
        qpos = qpos.at[:, 2].set(self.stand_z + h_off)
        qpos = qpos.at[:, self.act2jnt].set(_base + q_off)
        qvel = jnp.zeros((n, self.nv))
        qvel = qvel.at[:, 0].set(cy * vx - sy * vy)
        qvel = qvel.at[:, 1].set(sy * vx + cy * vy)
        qvel = qvel.at[:, 5].set(vyaw)
        qvel = qvel.at[:, self.act2vel].add(jv)
        cmd = jnp.stack([jax.random.uniform(rng, (n,), minval=self.cfg.cmd_vx_lo,
                                            maxval=self.cfg.cmd_vx_hi),
                         jnp.zeros(n)], axis=-1)
        data = self._empty_data.replace(qpos=jax.lax.stop_gradient(qpos),
                                        qvel=jax.lax.stop_gradient(qvel))
        rng, k3 = jax.random.split(rng)
        dr = {
            "kp_leg": jax.random.uniform(jax.random.fold_in(k3, 0), (n,), minval=self.cfg.dr_kp_leg_lo, maxval=self.cfg.dr_kp_leg_hi),
            "kd_leg": jax.random.uniform(jax.random.fold_in(k3, 1), (n,), minval=self.cfg.dr_kd_leg_lo, maxval=self.cfg.dr_kd_leg_hi),
            "kp_wheel": jax.random.uniform(jax.random.fold_in(k3, 2), (n,), minval=self.cfg.dr_kp_wheel_lo, maxval=self.cfg.dr_kp_wheel_hi),
            "tclip_leg": jax.random.uniform(jax.random.fold_in(k3, 3), (n,), minval=self.cfg.dr_tclip_lo, maxval=self.cfg.dr_tclip_hi),
            "tclip_wheel": jax.random.uniform(jax.random.fold_in(k3, 4), (n,), minval=self.cfg.dr_tclip_wheel_lo, maxval=self.cfg.dr_tclip_wheel_hi),
            "push_angle": jax.random.uniform(jax.random.fold_in(k3, 5), (n,), minval=0.0, maxval=2.0*np.pi),
        }
        return {
            "data": data,
            "rng": rng,
            "dr": dr,
            "ep_len": jnp.zeros(n, dtype=jnp.int32),
            "last_action": jnp.zeros((n, 16)),
            "cmd": cmd,
            "prev_progress": jnp.maximum(0.0, sy - self.start_y),   # init to spawn progress (no free bonus, legged_gym convention)
            "riser_crossed": jnp.zeros(n, dtype=jnp.int32),
            "done": jnp.zeros(n, dtype=bool),
            "success": jnp.zeros(n, dtype=bool),
        }