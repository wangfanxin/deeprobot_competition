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
        # v971: 防过伸低通滤波（消除 200Hz bang-bang 振荡）
        self._ov_sw_f = np.zeros(4)
        self._ov_st_f = np.zeros(4)
        # v984: swing 目标升速限制跟踪（防 PD 过冲悬空）
        self._sw_zt = None
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
            _dhmin = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
            if _dhv <= _dhmin:
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
        # v965: 诊断——每轮到棱距离/台面顶/完成标志（STAIRDBG 用）
        self._dbg_d = d.copy()
        self._dbg_top = top.copy()
        self._dbg_done = self._done.copy() if hasattr(self, "_done") else np.zeros(4)
        # 远离当前 riser 时复位完成标志（下一级）
        for i in range(4):
            if d[i] < -0.5:
                self._done[i] = False
        # v983: 轴式调度(用户批准方案)——前轴(FL+FR)成对触发、成对释放，
        # 后轴(RL+RR)等前轴全部 done 再触发。对称爬升消除单轮序列的贴面
        # 力不对称→自旋(v968/980/982 贴面前驱连续失败实测)。2 点支撑期
        # (前轴离地)roll 由 QP 差载控制(同轴两轮 y±0.181 可产生 roll 力矩)。
        for i in range(4):
            _front_axle = i in (0, 1)
            _rear_need = (bool(np.all(self._done[0:2]))
                          if not _front_axle else True)
            if self._sp[i] <= 0.0:
                _win_lo = -_swd
                _win_hi = 0.05
                _ok = (_win_lo < d[i] < _win_hi
                       and not self._done[i]
                       and _rear_need)
                if _ok:
                    if float(os.environ.get("S10_QP_DEBUG", "0")) > 2:
                        print('[PHASE] t=%.2f trig leg=%d d=%.3f top=%.3f '
                              'wz=%.3f swd=%.2f done=%s body=%.2f,%.2f'
                              % (self._t, i, d[i], top[i], wz[i], _swd,
                                 str(self._done), body_pos[0], body_pos[1]),
                              flush=True)
                    self._sp[i] = 1.0
                    self._sp_top[i] = top[i]
                    self._rel_t[i] = None
                    self._sw_t0[i] = self._t
                    # v984: 触发时目标跟踪从当前轮高开始
                    if getattr(self, '_sw_zt', None) is None:
                        self._sw_zt = np.zeros(4)
                    self._sw_zt[i] = float(wz[i])
            else:
                # v966: 释放放宽到轮心过棱(d>0.02)——原 d>0.08 让轮在台面
                # 上方悬空 2-3cm 等 d 推进(无抓地、狗不前进 3-4s 实测)；
                # 释放后支撑分支会压台面 2mm 恢复抓地
                if (d[i] > 0.02 and wz[i] >= self._sp_top[i] + r - 0.005
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
        """SWING 腿 place_z（沿 riser 立面 smoothstep，窗=SWING_D）。

        v976: 目标全部用 stair_world 几何，绝不回退到 lidar terrain_h——
        原选择窗 (-cl, 0.05) 之外(d<-0.12 或 d>0.05)pz=0 → swing 目标回退
        到 terrain_h+r，lidar 在棱口读 0.775 → 目标 0.856 把前轮抬到
        0.86-0.89 悬空、狗后仰翻车实测。
        """
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
            # v976: 找最近高 riser（无窗口限制）——任何 d 都有几何目标
            _best_d = 1e9
            _best = None
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                _dhmin2 = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
                if _dhv <= _dhmin2:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))
                if abs(_dd) < abs(_best_d):
                    _best_d = _dd
                    _best = (_rp, _tng, _dhv, _top)
            if _best is None:
                continue
            (_rp, _tng, _dhv, _top) = _best
            _z_bot = float(_top - _dhv)
            _d_w = _best_d
            # v928/v939: 贴面轮廓——d∈[-0.08,0] ramp(棱口才抬)，d<0.08
            # 几何地面，d>0 台面顶+r。
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

    # ---------------- 闭式 2 连杆 IK（v979） ----------------
    def _ik_closed(self, xd, zd):
        """给定 body 系轮心目标 (xd=前向, zd=向下为正)，返回 (q1, q2)。

        替代 8 次牛顿迭代——迭代从当前(折叠)姿态出发会收敛到"向上折叠"
        错误分支(q2≈2.8 轮越过髋、悬空 0.9+ 实测)。闭式解取 q2=+acos
        (自然膝弯曲分支)，一次到位。
        """
        L1, L2 = self.fk.L1, self.fk.L2
        r2 = xd * xd + zd * zd
        r2 = min(r2, (L1 + L2) ** 2 - 1e-6)
        c2 = float(np.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2),
                           -1.0, 1.0))
        q2 = float(np.arccos(c2))
        q1 = float(np.arctan2(xd, zd) - np.arctan2(
            L2 * np.sin(q2), L1 + L2 * np.cos(q2)))
        return q1, q2

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
            # v964: 抬升腿动态反作用力矩建模——SWING 腿拉轮向上，反作用
            # (向下)压在对应髋关节。把该外力矩加进 b0，QP 才能把载荷主动
            # 分配到支撑三角形(配合重心解 λ_ref)。大小 ∝ 抬升误差(目标轮高
            # -实际轮高)，方向沿世界 y(roll)/x(pitch)分量。
            try:
                _sw_ff = float(os.environ.get("S10_QP_SW_FF", "1.0"))
                if _sw_ff > 0.0:
                    _n_swx = int(np.sum(np.asarray(stance_mask) <= 0.5))
                    if _n_swx == 1:
                        _swi = int(np.argmax(np.asarray(stance_mask) <= 0.5))
                        _zt_sw = 0.0
                        if getattr(self, '_sw_z_tgt', None) is not None:
                            _zt_sw = float(self._sw_z_tgt[_swi])
                        _za_sw = float(wheel_xyz[_swi, 2])
                        _err_sw = max(0.0, _zt_sw - _za_sw)
                        if _err_sw > 0.0:
                            _kps_u = float(os.environ.get("S10_QP_KP_SW",
                                                          "100.0"))
                            if _swi in (2, 3):
                                _kps_u = float(os.environ.get(
                                    "S10_QP_KP_SW_REAR", "100.0"))
                            _f_sw = _sw_ff * _kps_u * _err_sw
                            _sx = 0.2277 if _swi in (0, 1) else -0.2277
                            _sy = 0.181 if _swi in (0, 2) else -0.181
                            _hip_sw = body["pos"] + body["R"] @ np.array(
                                [_sx, _sy, 0.0])
                            _rcs = _hip_sw - body["pos"]
                            _ixx = float(self.I_body[0, 0])
                            _iyy = float(self.I_body[1, 1])
                            # M = rc x (0,0,-f): Mx=-rc_y*f, My=+rc_x*f
                            b0[3] += (-_rcs[1] * _f_sw) / _ixx
                            b0[4] += (_rcs[0] * _f_sw) / _iyy
            except Exception:
                pass
            # a_des：z 高度阻尼 + roll/pitch 回零（世界系角加速度近似）
            a_des = np.zeros(6)
            # v973: z 参考自适应支撑高度——前轮上平台后轮心应在台面顶+r，
            # body 固定 0.78 时前腿只剩 3cm 垂距够不到台面，轮悬空 2-4cm
            # 无抓地、狗不推进(v971/972 实测)。z_ref = 各轮支撑轮心均值
            # + 腿垂距(drop)，支撑轮心用 stair_world 几何台面封顶(不用
            # 噪声 lidar)。全平:0.62+0.16=0.78；前上台面:0.844；全上台面:
            # 0.907。
            _z_ref = 0.78
            try:
                _sup = np.zeros(4)
                for _i in range(4):
                    _sup[_i] = float(terrain_h[_i]) + self.fk.r
                    _gtz = 0.0
                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        _dhmin3 = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
                        if _dhv <= _dhmin3:
                            continue
                        _ddz = float(np.dot(wheel_xyz[_i, :2] - _rp, _tng))
                        if _ddz > 0.0:
                            _gtz = max(_gtz, float(_top))
                    if _gtz > 0.4:
                        # v977: 过棱轮用几何台面(不取 min)——lidar 读低
                        # 会把 z_ref 拽下去
                        _sup[_i] = _gtz + self.fk.r
                _z_ref = float(np.mean(_sup)) + float(os.environ.get(
                    "S10_QP_Z_DROP", "0.16"))
            except Exception:
                pass
            self._z_ref = _z_ref
            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - _z_ref)
            _rear_sw = float(getattr(self, '_rear_swing', 0.0)) > 0.5
            # v970: roll/pitch 修正按 SWING 模式(step_lift)激活，不是接触
            # 掩码——贴面期 SWING 轮还在地上(qp_stance=1)，用接触掩码会
            # 把姿态控制全程关掉(v969 无效改动实测)。前轮单轮抬升期 body
            # 无俯仰控制 → 后仰 0.8-1.0rad；现在有接触感知+重心解+反作用
            # 力矩，QP 有可行域做修正。
            _any_sw_q = float(np.max(getattr(self, '_step_lift_last',
                                             np.zeros(4)))) > 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0"))
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0"))
            if not _any_sw_q:
                _ar_k = 0.0; _ap_k = 0.0
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
            _lamw = float(os.environ.get("S10_QP_LAM_W", "0.05"))
            P = A.T @ W1 @ A + _lamw * np.eye(n)
            q = -A.T @ W1 @ (a_des - b0) - _lamw * lam_ref.reshape(-1)
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
        self._body_roll = body["roll"]
        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)
        self._step_lift_last = step_lift.copy()
        # v997: 贴面区判定(软跟随与弱前驱共用)
        _face_drive = np.zeros(4, dtype=bool)
        try:
            _fd_lo = float(os.environ.get("S10_QP_FACE_DRIVE_LO", "-0.12"))
            _fd_hi = float(os.environ.get("S10_QP_FACE_DRIVE_HI", "0.05"))
            for _fl in range(4):
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    _dhmin = float(os.environ.get(
                        "S10_STAIR_RISER_MIN", "0.085"))
                    if _dhv <= _dhmin:
                        continue
                    _ddw = float(np.dot(wheel_xyz[_fl, :2] - _rp, _tng))
                    if _fd_lo < _ddw < _fd_hi:
                        _face_drive[_fl] = True
                        break
        except Exception:
            pass
        _fd_tau = -float(os.environ.get("S10_QP_FACE_DRIVE", "4.0"))
        self._rear_swing = float(np.max(step_lift[2:4])) > 0.5
        self._any_swing = float(np.max(step_lift)) > 0.5
        tau = np.zeros(16, dtype=np.float64)
        # λ_ref：支撑载荷基准——单轮抬升时用支撑三角形重心解(CoM 在三角
        # 形内的静态分配)，其余 mg/4 均载。3 点支撑下均载是错误目标：
        # RR 抬起时静解 FR≈RL≈mg/2、FL≈0，均载会让 QP 的载荷分布远离
        # 物理可行域(配合 b0 外力矩让姿态修正不饱和)。
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        # v967: QP 支撑掩码改为接触感知——SWING 轮只要还在地上(未离地)就
        # 继续算支撑(mg/4 均载)，只有真正离地才移除 QP 支撑并切换重心解。
        # 原因：SWING 触发(棱前 0.12m)时轮还在平地上，立即移除支撑让 QP 按
        # 3 点支撑分配(FR≈93/RL≈89/RR=10)在平地上把狗顶歪→向西漂移实测。
        qp_stance = np.ones(4)
        for _i in range(4):
            if step_lift[_i] > 0.5:
                _lo_z = float(terrain_h[_i]) + self.fk.r + 0.015
                if float(wheel_xyz[_i, 2]) > _lo_z:
                    qp_stance[_i] = 0.0
                # 轮还在地上 → 保持支撑（v967b: 原从 stance_mask 复制导致
                # SWING 轮永远 0，接触感知从未生效——必须在原地轮时置回 1）
        _sw_l = [i for i in range(4) if qp_stance[i] <= 0.5]
        if len(_sw_l) == 2 and sorted(_sw_l) in ([0, 1], [2, 3]):
            # v983: 轴式抬升 2 点支撑——同轴两轮均分 mg
            for _i in range(4):
                if qp_stance[_i] > 0.5:
                    lam_ref[_i, 2] = self.m * self.g / 2.0
        elif len(_sw_l) == 1:
            _st_l = [i for i in range(4) if qp_stance[i] > 0.5]
            _pts = []
            try:
                for _i in _st_l:
                    _pw = wheel_xyz[_i, :2] - body["pos"][:2]
                    _pb = R[:2, :2].T @ _pw
                    _pts.append(_pb)
                _Aq = np.array([[1.0, 1.0, 1.0],
                                [_pts[0][0], _pts[1][0], _pts[2][0]],
                                [_pts[0][1], _pts[1][1], _pts[2][1]]])
                _bq = np.array([1.0, 0.0, 0.0])
                _wq = np.linalg.solve(_Aq, _bq)
                _wq = np.clip(_wq, 0.0, 1.0)
                _ws = float(np.sum(_wq))
                if _ws > 1e-6:
                    _wq = _wq / _ws
                for _k, _i in enumerate(_st_l):
                    lam_ref[_i, 2] = float(_wq[_k]) * self.m * self.g
            except Exception:
                for _i in _st_l:
                    lam_ref[_i, 2] = self.m * self.g / 3.0
        else:
            for i in range(4):
                if qp_stance[i] > 0.5:
                    lam_ref[i, 2] = self.m * self.g / 4.0
        self._yaw_rate = float(qvel[5])
        lam = self._qp_solve(body, wheel_xyz, qp_stance, lam_ref)
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
                # v972: Jᵀλ 力项缩放——爬升期 QP 支撑力与位置 PD 打架
                # (前轮被泵到 0.9-1.1、body 后仰实测)。位置环主导时力项
                # 只做辅助载荷分配，防止过伸。
                _fsc = float(os.environ.get("S10_QP_FORCE_SCALE", "0.5"))
                th *= _fsc
                tk *= _fsc
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
                        # v959: 支撑腿目标压入台面 2mm（原 +0.01 余量让轮
                        # 悬空 0.01-0.06 无抓地、狗不推进实测）
                        # v977: 过棱后纯几何目标——lidar 在棱口读高(0.79)读低
                        # (0.44)都会把目标带偏(0.87/0.52)，轮已在台面上
                        _wzt = float(_gt_hi) + self.fk.r - 0.002
                        _hip_w2 = body["pos"] + R @ np.array(
                            [0.2277 if leg in (0, 1) else -0.2277, 0.0, 0.0])
                        _relx = (np.cos(yaw) * (wheel_xyz[leg, 0] - _hip_w2[0])
                                 + np.sin(yaw) * (wheel_xyz[leg, 1] - _hip_w2[1]))
                        _relz = float(np.clip(_wzt - _hip_w2[2], -0.36, 0.05))
                        # v979: 闭式 IK——迭代式从折叠姿态出发收敛到错误分支
                        _zd_t = max(0.001, -_relz)
                        _q1t, _q2t = self._ik_closed(_relx, _zd_t)
                        _q1t = float(np.clip(_q1t, -1.7, 1.0))
                        _q2t = float(np.clip(_q2t, -1.0, 3.0))
                        th += (_kpp * (_q1t - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (_q2t - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                        if float(os.environ.get("S10_QP_DEBUG", "0")) > 2:
                            p2 = self.fk.wheel_pos(_q1t, _q2t)
                            print('[STIK2] t=%.2f leg=%d q1t=%.2f q2t=%.2f '
                                  'errx=%.3f errz=%.3f th=%.1f tk=%.1f'
                                  % (self._t, leg, _q1t, _q2t,
                                     _relx - p2[0], _relz + p2[1], th, tk),
                                  flush=True)
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
                # v955: 支撑腿防过伸——轮在台面上时实际高度超过 geo-top+r
                # 过多(>0.03)就伸膝压回台面（FR 姿态期持续 1.03-1.06 悬空
                # 实测，stance PD 被 body roll 带高拉不回）
                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    # v971: 死区(0.01)+低通滤波——原 K=1000 全幅反馈在
                    # 200Hz 下 bang-bang 振荡(±48 高频抖动，轮悬空乱颤)
                    _db = float(os.environ.get("S10_QP_OV_DB", "0.010"))
                    _ov_des = 0.0
                    if _over_s > _db:
                        _ov_des = float(os.environ.get(
                            "S10_QP_K_OVER_ST", "1000.0")) * (_over_s - _db)
                    _lp = float(os.environ.get("S10_QP_OV_LP", "0.25"))
                    self._ov_st_f[leg] += _lp * (_ov_des - self._ov_st_f[leg])
                    # v987: 经 pz 梯度方向(同 swing)——折叠位姿下满幅压轮
                    _Jz2 = np.array([J[1, 0], J[1, 1]])
                    _nz2 = float(np.linalg.norm(_Jz2)) + 1e-6
                    _ovg2 = float(np.clip(self._ov_st_f[leg], -48.0, 48.0))
                    _tov2 = _Jz2 / _nz2 * _ovg2
                    th += float(_tov2[0])
                    tk += float(_tov2[1])
                    if float(os.environ.get("S10_QP_DEBUG", "0")) > 2:
                        _hipz3 = float(_hip_w2[2])
                        print('[PLANT] t=%.2f leg=%d q1=%.2f q2=%.2f '
                              'q1t=%.2f q2t=%.2f bz=%.3f hipz=%.3f '
                              'wz=%.3f wzt=%.3f ovf=%.1f th=%.1f tk=%.1f'
                              % (self._t, leg, q1, q2, _q1t, _q2t,
                                 body["pos"][2], _hipz3, wheel_xyz[leg, 2],
                                 _wzt, self._ov_st_f[leg], th, tk),
                              flush=True)
                    if float(os.environ.get("S10_QP_DEBUG", "0")) > 2:
                        print('[OVST] t=%.2f leg=%d wz=%.3f top=%.3f '
                              'ov=%.3f ovf=%.1f tk->%.1f'
                              % (self._t, leg, wheel_xyz[leg, 2], _gt_hi,
                                 _over_s, self._ov_st_f[leg], tk), flush=True)
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
                # v964: 记录抬升目标轮高（QP 外力矩建模用）
                if getattr(self, '_sw_z_tgt', None) is None:
                    self._sw_z_tgt = np.zeros(4)
                self._sw_z_tgt[leg] = _wz_t
                # v984: 目标升速限制——贴面轮廓随前速变化可达 3m/s，PD 追
                # 不上→滞后积累后过冲(轮悬空 0.75-0.9 无抓地实测)。目标只
                # 允许升 SW_TGT_RATE(默认1.2m/s)，轮靠贴面接触自然上升，
                # PD 只做引导；目标可自由下降(取 min)。
                _sw_rate = float(os.environ.get("S10_QP_SW_TGT_RATE", "1.2"))
                if getattr(self, '_sw_zt', None) is None:
                    self._sw_zt = np.zeros(4)
                _wz_t = min(_wz_t, self._sw_zt[leg] + _sw_rate * dt)
                self._sw_zt[leg] = _wz_t
                _hip_w = body["pos"] + R @ np.array(
                    [0.2277 if leg in (0, 1) else -0.2277, 0.0, 0.0])
                _rel = np.array([np.cos(yaw) * (wheel_xyz[leg, 0] - _hip_w[0])
                                 + np.sin(yaw) * (wheel_xyz[leg, 1] - _hip_w[1]),
                                 _wz_t - _hip_w[2]])
                _rz = float(np.clip(_rel[1], -0.34, 0.15))
                # v979: 闭式 IK(同支撑腿)——迭代式易收敛到向上折叠分支
                _zd_s = max(0.001, -_rz)
                q1t, q2t = self._ik_closed(float(_rel[0]), _zd_s)
                # v929: 防镜像折叠过伸(下界 -1.1 保留)；上界放宽
                q1t = float(np.clip(q1t, -1.1, 0.5))
                q2t = float(np.clip(q2t, -1.0, 3.0))
                # v934: 前后轴抬升增益不对称——前轮爬升有动量辅助用软增益
                # （防过伸/泵高）；后轮爬顶需主动抬升 0.125m 用硬增益
                _kps_d = float(os.environ.get("S10_QP_KP_SW", str(self.kp)))
                if leg in (2, 3):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_REAR",
                                                  str(self.kp)))
                # v935: swing kd 默认 30（原 8 阻尼比 ~0.25 欠阻尼，轮离地
                # 后自由过冲到 1.1 悬空实测）
                _kds_d = float(os.environ.get("S10_QP_KD_SW", "30.0"))
                # v997: 贴面软阻尼跟随——前轮贴面时 KP 降到 15、KD 提到 100，
                # 目标跟轮高+3mm，轮靠前驱滚上立面，腿只吸收冲击。
                if _face_drive[leg] and leg in (0, 1):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_SOFT", "15.0"))
                    _kds_d = float(os.environ.get("S10_QP_KD_SW_SOFT", "100.0"))
                    _wz_t = float(wheel_xyz[leg, 2]) + float(os.environ.get(
                        "S10_QP_FOLLOW_GAP", "0.003"))
                tau[hipy_i] = (_kps_d * (q1t - q1)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kps_d * (q2t - q2)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                # v952: 防过伸反馈——swing 轮实际高度超过目标>0.02m 时，
                # 膝盖加强制伸展力矩把轮压回台面（过伸是动力学折叠惯性，
                # 位置 PD 拉不回：单轮序列 FR 冲到 1.08 实测）。减少 q2
                # (伸膝) 推轮向下。
                if float(os.environ.get("S10_QP_DEBUG", "0")) > 2:
                    print('[SWDBG] t=%.2f leg=%d q1=%.2f q2=%.2f q1t=%.2f '
                          'q2t=%.2f wz=%.3f wzt=%.3f bz=%.3f relx=%.2f '
                          'tauH=%.1f tauK=%.1f'
                          % (self._t, leg, q1, q2, q1t, q2t,
                             wheel_xyz[leg, 2], _wz_t, body["pos"][2],
                             float(_rel[0]), tau[hipy_i], tau[knee_i]),
                          flush=True)
                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                _db2 = float(os.environ.get("S10_QP_OV_DB_SW", "0.020"))
                _ov2_des = 0.0
                if _over2 > _db2:
                    _k_ov = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                    _ov2_des = _k_ov * (_over2 - _db2)
                _lp2 = float(os.environ.get("S10_QP_OV_LP_SW", "0.25"))
                self._ov_sw_f[leg] += _lp2 * (_ov2_des - self._ov_sw_f[leg])
                # v987: 防过伸沿 pz 增大梯度方向施加满力矩——J^T 投影在折叠
                # 位姿(q1+q2≈π)下近奇异，45N 力只剩 ~5Nm 关节力矩推不动轮
                # (v985/986 实测轮仍悬空 5cm+)。归一化梯度任何姿态都满幅。
                if abs(self._ov_sw_f[leg]) > 0.5:
                    _Jz = np.array([J[1, 0], J[1, 1]])
                    _nz = float(np.linalg.norm(_Jz)) + 1e-6
                    _ovg = float(np.clip(self._ov_sw_f[leg], -48.0, 48.0))
                    _tov = _Jz / _nz * _ovg
                    tau[hipy_i] += float(_tov[0])
                    tau[knee_i] += float(_tov[1])
                if float(os.environ.get("S10_QP_DEBUG", "0")) > 2:
                    print('[OVSW] t=%.2f leg=%d wz=%.3f tgt=%.3f ov=%.3f '
                          'ovf=%.1f tauK->%.1f'
                          % (self._t, leg, wheel_xyz[leg, 2], _wz_t,
                             _over2, self._ov_sw_f[leg], tau[knee_i]),
                          flush=True)
        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        for leg in range(4):
            _wq = float(qvel[WHEEL_QV_IDX[leg]])
            _vw = -_wq * self.fk.r
            if stance_mask[leg] > 0.5:
                if _any_swing:
                    # v942: 任何 swing 期支撑轮速度 PID（v958 开环满驱在前
                    # 轮爬升期自旋 3s 翻车回退）
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    # v999: 任意 swing 期差速抗旋(前轮抬空失去 yaw 阻力)
                    _om_gain = float(qvel[5]) * _kd_y
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    if _face_drive[leg] and float(wheel_xyz[leg, 2]) < 0.72:
                        # v998: 轮低于 0.72(贴面爬升中)才前驱，过顶即停——
                        # 前驱持续把轮顶到 1.2+ 过伸(v997 实测)
                        _tw = min(float(_tw), _fd_tau)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
                else:
                    # v981: 全支撑(接近段)执行导航 omega 差速——原仅跟 vx_f
                    # 导致进梯前 yaw 漂移 0.2-0.5rad，首轮贴面不对称→自旋
                    _vref = (self._vx_f
                             - _side_s[leg] * self._om_f * self.track_half)
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = (_fd_tau if (_face_drive[leg]
                                         and float(wheel_xyz[leg, 2]) < 0.72)
                                         else -1.5)
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
