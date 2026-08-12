"""StairWBC-QP：位置基 + QP 力分配主环（v906，终版核心）。

解决位置基 FP 在爬顶的结构性失败（body 高度与后轮行程矛盾）：
- ModeSchedule 布尔相位（同 StairWBC）；
- SWING 腿：位置控制（贴面爬升 IK，v899 加宽窗 + v904 膝角放宽）；
- STANCE 腿：接触力由 osqp 12 变量 QP 解出（SRBD：a_base = A·λ + b0），
  同时满足 body 姿态（z/pitch/roll 阻尼）与后轮静压（λ_z ≥ N_min）；
- 轮矩：STANCE 前驱、SWING 0（轮差速冻结，hip yaw 全程激活）。

数据管线（200Hz）：
  qpos/qvel/wheel_xyz + stair_world -> ModeSchedule -> a_des + λ_ref
  -> QP(λ, 12var, osqp<2ms) -> τ_leg = Jᵀ·λ(支撑) / IK PD(抬升)
  -> τ_wheel(前驱/0) -> 16 维力矩
"""
import os

import numpy as np

from .stair_vmc_legs import (S10LegFK, LEG_CTRL_IDX, LEG_Q_IDX, LEG_QV_LEG,
                             WHEEL_Q_IDX, WHEEL_QV_IDX, WHEEL_BODY)


class StairWBCQP:
    def __init__(self, mass=19.0, g=9.81, L1=0.18, L2=0.18, r=0.081,
                 track_half=0.24, kp=200.0, kd=8.0,
                 wheel_k=10.0, wheel_d=0.02):
        self.m, self.g = mass, g
        self.fk = S10LegFK(L1, L2, r)
        self.track_half = track_half
        self.kp = kp
        self.kd = kd
        self.wheel_k = wheel_k
        self.wheel_d = wheel_d
        # body 惯量近似（kg·m²）
        self.I_body = np.diag([0.15, 0.25, 0.15])
        self.stair_world = []
        self.stair = None
        # v910: 支撑腿位置保持目标（STAND 半蹲，与主脚本 STAND_TARGET 一致）
        self.pose_target = np.array([-0.05, -1.10, 1.90,
                                      0.05, -1.10, 1.90,
                                     -0.05,  1.10, -1.90,
                                      0.05,  1.10, -1.90],
                                    dtype=np.float64)
        # ModeSchedule 状态
        self._sp_f = 0.0
        self._sp_r = 0.0
        self._sp_f_top = 0.0
        self._sp_r_top = 0.0
        self._rel_f_t = None
        self._rel_r_t = None
        self._sw_f_t0 = -1e9
        self._sw_r_t0 = -1e9
        self._t = 0.0
        self._vx_f = 0.0
        self._om_f = 0.0
        self._osqp = None
        try:
            import osqp
            self._osqp = osqp
        except Exception:
            self._osqp = None

    # ---------------- ModeSchedule（同 StairWBC） ----------------
    def _nearest_riser(self, ax):
        dmin, top = 1e9, 0.0
        for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
            if _dhv <= 0.085:
                continue
            _dd = float(np.dot(np.asarray(ax, dtype=np.float64) - _rp, _tng))
            if abs(_dd) < abs(dmin):
                dmin, top = _dd, float(_top)
        return dmin, top

    # v949: 单轮序列 ModeSchedule——FL->FR->RL->RR 依次爬升，任意时刻
    # >=3 点支撑（消除 2 点支撑 roll 崩）。每轮独立相位：
    #   触发: d_i 进入窗内（FL/RL 提前 0.30，FR/RR 等对侧爬完）
    #   释放: d_i>0.08 且轮高>=台面顶+r
    #   序列约束: FR 等 FL 完成; RL 等前轴完成; RR 等 RL 完成
    def _update_phases(self, body_pos, fwd, wheel_xyz):
        r = self.fk.r
        _swd = float(os.environ.get("S10_STAIR_SWING_D", "0.30"))
        _to = float(os.environ.get("S10_STAIR_SWING_TO", "1.5"))
        if not hasattr(self, "_sp"):
            self._sp = np.zeros(4)
            self._sp_top = np.zeros(4)
            self._sw_t0 = np.full(4, -1e9)
            self._rel_t = [None] * 4
            self._done = np.zeros(4, dtype=bool)
        d = np.zeros(4); top = np.zeros(4)
        for i in range(4):
            d[i], top[i] = self._nearest_riser(wheel_xyz[i, :2])
        wz = wheel_xyz[:, 2]
        # 远离当前 riser 时复位完成标志（下一级）
        for i in range(4):
            if d[i] < -0.5:
                self._done[i] = False
        # 触发/释放
        for i in range(4):
            _lead = (i in (0, 2))           # FL/RL 提前触发
            _opp_done = self._done[i ^ 1]   # 对侧轮完成（FR 等 FL, RR 等 RL）
            _front_done = bool(np.all(self._done[0:2])) if i >= 2 else True
            if self._sp[i] <= 0.0:
                _win_lo = -_swd if _lead else -0.05
                _win_hi = 0.05 if _lead else 0.10
                _ok = (_win_lo < d[i] < _win_hi
                       and not self._done[i]
                       and (not _lead or _front_done)
                       and (_lead or _opp_done))
                if _ok:
                    self._sp[i] = 1.0
                    self._sp_top[i] = top[i]
                    self._rel_t[i] = None
                    self._sw_t0[i] = self._t
            else:
                if (d[i] > 0.08 and wz[i] >= self._sp_top[i] + r + 0.005
                        and not self._done[i]):
                    self._done[i] = True
                    if self._rel_t[i] is None:
                        self._rel_t[i] = self._t
                    elif self._t - self._rel_t[i] >= 0.05:
                        self._sp[i] = 0.0
                        self._rel_t[i] = None
                else:
                    self._rel_t[i] = None
                if self._t - self._sw_t0[i] > _to:
                    self._sp[i] = 0.0
                    self._rel_t[i] = None
        step_lift = self._sp.copy()
        place_z = np.array([self._sp_top[i] if self._sp[i] > 0 else 0.0
                            for i in range(4)], dtype=np.float64)
        return step_lift, place_z

    # ---------------- 贴面爬升 SWING 位置目标 ----------------
    def _face_place_z(self, wheel_xyz, step_lift):
        """SWING 腿 place_z（沿 riser 立面 smoothstep，窗=SWING_D）。"""
        pz = np.zeros(4)
        _r = self.fk.r
        _cl = float(os.environ.get("S10_STAIR_SWING_D", "0.30"))
        _margin = float(os.environ.get("S10_STAIR_LIFT_MARGIN", "0.04"))
        for _leg in range(4):
            if step_lift[_leg] <= 0.02:
                continue
            # v949: 单轮序列——贴面目标用该轮自身位置（不再用轴均值，
            # 否则左右轮同相抬升破坏序列）
            _ax_xy = wheel_xyz[_leg, :2]
            _best_d = 1e9
            _best = None
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                if _dhv <= 0.085:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))
                if -_cl < _dd < 0.05 and abs(_dd) < abs(_best_d):
                    _best_d = _dd
                    _best = (_rp, _tng, _dhv, _top)
            if _best is None:
                continue
            (_rp, _tng, _dhv, _top) = _best
            _z_bot = float(_top - _dhv)
            _d_w = float(np.dot(_ax_xy - _rp, _tng))
            # v928: 贴面轮廓（v901 同款，符号修正）——d<0=棱前：
            # d∈[-cl,0] 平滑 ramp flat→top（d=0 到台面顶），d>0 台面顶+r。
            # 此前 v920/v924 把符号写反（d>0 过棱给平地，d<-0.05 反而给
            # 台面顶），前轮抬不上去/过伸实测。
            # v939: 抬升窗收紧到 [-0.08, 0]（轮半径 0.081，d=-0.08 时轮
            # 正好贴棱）——v901 的 [-cl,0] 窗在轮还在地上 0.3m 时就开始抬
            # → 轮推不动反顶 body（0.96 泵高、后腿够不到、roll 崩实测）。
            # 棱口才抬，动量+贴面把轮带上去。
            if _d_w <= 0.0:
                if _d_w >= -0.08:
                    _t = float(np.clip((_d_w + 0.08) / 0.08, 0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    _z_face = min(_z_bot + _r + _dhv * _ss, _top + _r + 0.005)
                else:
                    _z_face = _z_bot + _r
            else:
                _z_face = min(_top + _r, _top + _r + 0.005)
            pz[_leg] = _z_face - _r - _margin
        return pz

    # ---------------- QP：接触力分配（SRBD） ----------------
    def _qp_solve(self, body, wheel_xyz, stance_mask, lam_ref):
        if self._osqp is None:
            return lam_ref
        try:
            from scipy import sparse
            n = 12
            m = self.m
            I_world = body["R"] @ self.I_body @ body["R"].T
            A = np.zeros((6, n))
            for i in range(4):
                rc = wheel_xyz[i] - body["pos"]
                A[0:3, i * 3:i * 3 + 3] = np.eye(3) / m
                rcm = np.array([[0, -rc[2], rc[1]],
                                [rc[2], 0, -rc[0]],
                                [-rc[1], rc[0], 0]])
                A[3:6, i * 3:i * 3 + 3] = np.linalg.inv(I_world) @ rcm
            b0 = np.zeros(6)
            b0[2] = -self.g
            # a_des：z 高度阻尼 + roll/pitch 回零（世界系角加速度近似）
            a_des = np.zeros(6)
            # v939 原状: z 参考固定 0.78（v944/v945 试验 z 跟随/0.88 均引发
            # roll 崩回退）——最佳状态为前轮上台面稳定 15s 卡死。
            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - 0.78)
            _rear_sw = float(getattr(self, '_rear_swing', 0.0)) > 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0")) if _rear_sw else 0.0
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0")) if _rear_sw else 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]
            # v947: yaw 率阻尼只在后轮爬顶期启用——前轮 SWING 期(双后腿
            # 支撑)QP 用侧向力阻尼 yaw 造成载荷极端不对称(RL 161N vs RR
            # 10N, 横向到摩擦锥极限 → roll 崩实测)；后轮爬顶期(双前腿)
            # 才有支撑做侧向力
            _ay_k = (float(os.environ.get("S10_QP_AY_K", "-20.0"))
                     if float(getattr(self, '_rear_swing', 0.0)) > 0.5
                     else 0.0)
            a_des[5] = _ay_k * getattr(self, '_yaw_rate', 0.0)
            W1 = np.diag([0.0, 0.0,
                          float(os.environ.get("S10_QP_W_Z", "400.0")),
                          float(os.environ.get("S10_QP_W_R", "20.0")),
                          float(os.environ.get("S10_QP_W_P", "20.0")),
                          float(os.environ.get("S10_QP_W_Y", "10.0"))])
            P = A.T @ W1 @ A + 1e-2 * np.eye(n)
            q = -A.T @ W1 @ (a_des - b0) - 1e-2 * lam_ref.reshape(-1)
            rows, cols, vals, lo, hi = [], [], [], [], []
            mus = float(os.environ.get("S10_QP_MU", "0.6"))
            nmin = float(os.environ.get("S10_QP_NMIN", "10.0"))
            for i in range(4):
                base = i * 3
                if stance_mask[i] <= 0.5:
                    for k in range(3):
                        rows.append(len(lo)); cols.append(base + k); vals.append(1.0)
                        lo.append(0.0); hi.append(0.0)
                else:
                    rows.append(len(lo)); cols.append(base + 2); vals.append(-1.0)
                    lo.append(-np.inf); hi.append(-nmin)
                    for k in (0, 1):
                        rows.append(len(lo)); cols.append(base + k); vals.append(1.0)
                        rows.append(len(lo)); cols.append(base + 2); vals.append(-mus)
                        lo.append(-np.inf); hi.append(0.0)
                        rows.append(len(lo)); cols.append(base + k); vals.append(-1.0)
                        rows.append(len(lo)); cols.append(base + 2); vals.append(-mus)
                        lo.append(-np.inf); hi.append(0.0)
            # v909: 总法向力支撑约束——QP 若只追姿态会把 λ_z 压到下限
            # 10N，狗失支撑坠落翻车（台架实测 λ_z=10 全轮）。硬约束
            # Σ λ_z(stance) ≥ 0.92mg，姿态修正只能在保支撑前提下做。
            for i in range(4):
                if stance_mask[i] > 0.5:
                    rows.append(len(lo)); cols.append(i * 3 + 2); vals.append(1.0)
            lo.append(self.m * self.g * 0.92)
            hi.append(np.inf)
            A_sp = sparse.csc_matrix(
                (np.asarray(vals, dtype=np.float64),
                 (np.asarray(rows, dtype=np.int64),
                  np.asarray(cols, dtype=np.int64))),
                shape=(len(lo), n))
            prob = self._osqp.OSQP()
            prob.setup(P=sparse.csc_matrix(P), q=q, A=A_sp,
                       l=np.asarray(lo), u=np.asarray(hi),
                       verbose=False, time_limit=0.002, eps_abs=1e-3,
                       eps_rel=1e-3, max_iter=400, polish=False)
            res = prob.solve()
            if float(os.environ.get('S10_QP_DEBUG', '0')) > 0:
                _st_ok = res.info.status in ('solved', 'solved inaccurate')
                _lam_s = (np.round(np.asarray(res.x, dtype=np.float64).reshape(4, 3), 2)
                          if _st_ok else np.zeros((4, 3)))
                print('[QP] t=%.2f st=%s ad=[%.2f %.2f %.2f %.2f %.2f %.2f] lam=%s st=%s'
                      % (self._t, str(stance_mask), a_des[0], a_des[1], a_des[2],
                         a_des[3], a_des[4], a_des[5], np.round(_lam_s, 2).tolist(),
                         res.info.status), flush=True)
            if res.info.status in ("solved", "solved inaccurate"):
                return np.asarray(res.x, dtype=np.float64).reshape(4, 3)
            return lam_ref
        except Exception:
            return lam_ref

    # ---------------- 主入口 ----------------
    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):
        self._t += dt
        q = qpos[3:7]
        w, x, y, z = q
        yaw = float(np.arctan2(2.0 * (w * z + x * y),
                               1.0 - 2.0 * (y * y + z * z)))
        roll = float(np.arctan2(2.0 * (w * x + y * z),
                                1.0 - 2.0 * (x * x + y * y)))
        pitch = float(np.arctan2(2.0 * (w * y - z * x),
                                 1.0 - 2.0 * (y * y + x * x)))
        R = np.asarray([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        body = dict(pos=qpos[0:3], yaw=yaw, roll=roll, pitch=pitch, R=R)
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        self._vx_f += (float(cmd.get("vx", 0.0)) - self._vx_f) * min(1.0, dt / 0.10)
        self._om_f += (float(cmd.get("omega", 0.0)) - self._om_f) * min(1.0, dt / 0.10)
        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)
        self._rear_swing = float(np.max(step_lift[2:4])) > 0.5
        self._any_swing = float(np.max(step_lift)) > 0.5
        tau = np.zeros(16, dtype=np.float64)
        # λ_ref：支撑 mg/4 均载，抬升 0
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        for i in range(4):
            if stance_mask[i] > 0.5:
                lam_ref[i, 2] = self.m * self.g / 4.0
        self._yaw_rate = float(qvel[5])
        lam = self._qp_solve(body, wheel_xyz, stance_mask, lam_ref)
        # 腿力矩：支撑 Jᵀλ（世界→body→矢状面），抬升 IK 位置 PD
        for leg in range(4):
            b = leg * 3
            q1 = float(qpos[LEG_Q_IDX[b + 1]])
            q2 = float(qpos[LEG_Q_IDX[b + 2]])
            qhx = float(qpos[LEG_Q_IDX[b]])
            J = self.fk.jac(q1, q2)
            hipx_i = LEG_CTRL_IDX[b]
            hipy_i = LEG_CTRL_IDX[b + 1]
            knee_i = LEG_CTRL_IDX[b + 2]
            # hipx 姿态保持（roll 修正）
            _q0_tgt = -0.05 if leg in (0, 1) else 0.05
            _kpx = float(os.environ.get("S10_QP_KP_HIPX", str(self.kp)))
            tau[hipx_i] = _kpx * (_q0_tgt - qhx) - self.kd * float(qvel[6 + LEG_QV_LEG[b]])
            if stance_mask[leg] > 0.5:
                # 接触力 λ（世界）→ body 系 → 矢状面 → Jᵀ
                f_w = lam[leg]
                f_b = R.T @ f_w
                f_s = np.array([float(f_b[0]), -float(f_b[2])])
                th, tk = J.T @ f_s
                # v940: 支撑腿台面感知位置保持——轮过了 riser 棱(d>0, 在
                # 台面上)时目标封顶到台面顶+r，让前轮在后轮爬顶期压住台面。
                # v936/937 失败是因为宽抬升窗泵高 body 后腿够不到；v939 窄
                # 窗已解决泵高，现在叠加。平地/接近段保持原逻辑。
                _kpp = float(os.environ.get("S10_QP_KP_POS", "80.0"))
                _kdp = float(os.environ.get("S10_QP_KD_POS", "6.0"))
                # v941: 支撑腿 IK 只在轮真正过了 riser 棱(d>0, 在台面上)
                # 时启用——v940 全时段 IK 在接近段与 λ 冲突振荡(多解)翻车
                # 实测；接近段保持姿势 PD(稳定)。
                _gt_hi = -1.0
                try:
                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        if _dhv <= 0.085:
                            continue
                        _dd = float(np.dot(wheel_xyz[leg, :2] - _rp, _tng))
                        if _dd > 0.0:
                            _gt_hi = max(_gt_hi, float(_top))
                except Exception:
                    pass
                if _gt_hi > 0.4:
                    try:
                        _wzt = min(float(terrain_h[leg]) + self.fk.r,
                                   _gt_hi + self.fk.r + 0.01)
                        _hip_w2 = body["pos"] + R @ np.array(
                            [0.2277 if leg in (0, 1) else -0.2277, 0.0, 0.0])
                        _relx = (np.cos(yaw) * (wheel_xyz[leg, 0] - _hip_w2[0])
                                 + np.sin(yaw) * (wheel_xyz[leg, 1] - _hip_w2[1]))
                        _relz = float(np.clip(_wzt - _hip_w2[2], -0.36, 0.05))
                        _q1t, _q2t = q1, q2
                        for _ in range(8):
                            _p = self.fk.wheel_pos(_q1t, _q2t)
                            _err = np.array([_relx - _p[0], _relz + _p[1]])
                            _Jj = self.fk.jac(_q1t, _q2t)
                            _dq = np.linalg.lstsq(_Jj, _err, rcond=None)[0]
                            _dq = np.clip(_dq, -0.2, 0.2)
                            _q1t += float(_dq[0]); _q2t += float(_dq[1])
                            _q1t = float(np.clip(_q1t, -1.7, -0.35))
                            _q2t = float(np.clip(_q2t, -0.2, 3.0))
                        th += (_kpp * (_q1t - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (_q2t - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                    except Exception:
                        th += (_kpp * (self.pose_target[b + 1] - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (self.pose_target[b + 2] - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                else:
                    th += (_kpp * (self.pose_target[b + 1] - q1)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                    tk += (_kpp * (self.pose_target[b + 2] - q2)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                tau[hipy_i] = float(np.clip(th, -48, 48))
                tau[knee_i] = float(np.clip(tk, -48, 48))
            else:
                # 抬升：位置 PD 到贴面目标（简单 IK）
                _sl = float(step_lift[leg])
                _pz = float(place_z[leg])
                _wz_t = float(terrain_h[leg]) + self.fk.r
                if _pz > 0.01 and _sl > 0.02:
                    # v950: swing 目标严格封顶到台面顶+r+0.005——原
                    # body_z+0.25 松上限允许轮抬到 1.08（FR/RL 过伸实测）；
                    # 台面顶来自 place_z 反解（pz = 顶-r-margin）
                    _wz_t = min(_pz + self.fk.r + 0.04,
                                _pz + self.fk.r + 0.045)
                _hip_w = body["pos"] + R @ np.array(
                    [0.2277 if leg in (0, 1) else -0.2277, 0.0, 0.0])
                _rel = np.array([np.cos(yaw) * (wheel_xyz[leg, 0] - _hip_w[0])
                                 + np.sin(yaw) * (wheel_xyz[leg, 1] - _hip_w[1]),
                                 _wz_t - _hip_w[2]])
                _rz = float(np.clip(_rel[1], -0.34, 0.15))
                q1t, q2t = q1, q2
                for _ in range(8):
                    p = self.fk.wheel_pos(q1t, q2t)
                    err = np.array([_rel[0] - p[0], _rz + p[1]])
                    Jj = self.fk.jac(q1t, q2t)
                    dq = np.linalg.lstsq(Jj, err, rcond=None)[0]
                    dq = np.clip(dq, -0.25, 0.25)
                    q1t += float(dq[0]); q2t += float(dq[1])
                    # v929: SWING 腿 q1 正常分支（防镜像折叠过伸，同 FP v925）
                    q1t = float(np.clip(q1t, -1.1, -0.3))
                    q2t = float(np.clip(q2t, 0.5, 3.0))
                # v934: 前后轴抬升增益不对称——前轮爬升有动量辅助用软增益
                # （防过伸/泵高）；后轮爬顶需主动抬升 0.125m 用硬增益
                _kps_d = float(os.environ.get("S10_QP_KP_SW", str(self.kp)))
                if leg in (2, 3):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_REAR",
                                                  str(self.kp)))
                # v935: swing kd 默认 30（原 8 阻尼比 ~0.25 欠阻尼，轮离地
                # 后自由过冲到 1.1 悬空实测）
                _kds_d = float(os.environ.get("S10_QP_KD_SW", "30.0"))
                tau[hipy_i] = (_kps_d * (q1t - q1)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kps_d * (q2t - q2)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 2]]))
        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        for leg in range(4):
            _wq = float(qvel[WHEEL_QV_IDX[leg]])
            _vw = -_wq * self.fk.r
            if stance_mask[leg] > 0.5:
                if _any_swing:
                    # v942: 任何 swing 期支撑轮开环满前驱（用户 A3）——
                    # 速度 PID 在爬升期把支撑轮刹车(+13.5 反向实测) → 狗
                    # 卡在棱前不前进、后轮够不到 SWING 窗。v931 失败时
                    # 是宽抬升窗+泵高环境，现 v939 窄窗+v935 阻尼已不同。
                    # 后轮爬顶期叠加 yaw 率阻尼差速抗自旋（v938）。
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = (float(qvel[5]) * _kd_y
                                if float(np.max(step_lift[2:4])) > 0.5 else 0.0)
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
                else:
                    _tw = -(self.wheel_k * (self._vx_f - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = -1.5
        # hip yaw（导航误差 + yaw 率阻尼）
        try:
            _kp_y = float(os.environ.get("S10_FP_YAW_KP", "2.0"))
            _kd_y = float(os.environ.get("S10_FP_YAW_KD", "0.5"))
            _yerr = 0.0
            if self.stair is not None:
                _yerr = float(getattr(self.stair, "_last_err", 0.0))
            _th_y = _kp_y * _yerr - _kd_y * float(qvel[5])
            tau[LEG_CTRL_IDX[0]] += _th_y
            tau[LEG_CTRL_IDX[3]] -= _th_y
            tau[LEG_CTRL_IDX[6]] += 0.5 * _th_y
            tau[LEG_CTRL_IDX[9]] -= 0.5 * _th_y
        except Exception:
            pass
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -48, 48)
        tau[WHEEL_Q_IDX] = np.clip(tau[WHEEL_Q_IDX], -13.5, 13.5)
        return tau
