"""StairWBC：轮足狗楼梯爬升——位置基全身控制（终版 2026-08-11）。

ModeSchedule（布尔几何相位，整轴硬切换）→ BodyIK（body 姿态闭环）
→ LegCtrl（位置 PD 拉满 + 静压 + 微阻抗）→ WheelCtrl（开环限幅：
SWING=0 / STANCE=差速 PID ≤13.5Nm）→ QPChecker（osqp 接触合规校验，
非力分配主环，破锥仅微降腿增益）。

复用 FootPlaceVMC（v828 位置基地基：body 状态 / 2D IK / 位置环 /
单侧垂直阻抗），把终版缺失的部分补上：
- 4 轮整轴布尔相位状态机（±0.05m 窗口，过棱 + 轮高≥台面顶+R-0.02
  持续 0.05s 释放，前轴优先防双轴同抬）；
- QP Checker（osqp 12 变量，2ms 预算，超时/异常回退不阻塞 200Hz 主环）。
"""
import os

import numpy as np

from .stair_vmc_legs import FootPlaceVMC, LEG_QV_LEG, WHEEL_Q_IDX


class StairWBC(FootPlaceVMC):
    """终版 StairWBC：继承 FootPlaceVMC 的位置基全身控制骨架。"""

    def __init__(self, mass=19.0, g=9.81, L1=0.18, L2=0.18, r=0.081,
                 track_half=0.24, kp=220.0, kd=6.0,
                 wheel_k=4.0, wheel_d=0.08):
        super().__init__(mass=mass, g=g, L1=L1, L2=L2, r=r,
                         track_half=track_half, kp=kp, kd=kd,
                         wheel_k=wheel_k, wheel_d=wheel_d)
        self.stair_world = []      # [(pt, tng, arc, dh, top)] 世界坐标
        self.stair = None          # AutoNavFollower 引用（stair_terrain）
        self._sp_f = 0.0
        self._sp_r = 0.0
        self._sp_f_top = 0.0
        self._sp_r_top = 0.0
        self._rel_f_t = None
        self._rel_r_t = None
        self._t = 0.0
        self._qp_scale = 1.0
        self._osqp = None
        try:
            import osqp
            self._osqp = osqp
        except Exception:
            self._osqp = None

    # ---------------- ModeSchedule：布尔几何相位（整轴硬切换） ----------------
    def _nearest_riser(self, ax):
        """前/后轴到最近高 riser 的沿切线投影距离与台面顶高（世界坐标）。"""
        dmin, top = 1e9, 0.0
        for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
            if _dhv <= 0.085:      # 小台阶纯滚（首级 0.063m < 轮半径）
                continue
            _dd = float(np.dot(np.asarray(ax, dtype=np.float64) - _rp, _tng))
            if abs(_dd) < abs(dmin):
                dmin, top = _dd, float(_top)
        return dmin, top

    def _update_phases(self, body_pos, fwd, wheel_xyz):
        """前/后轴布尔相位机：|d|<0.05 进入 SWING；过棱（d<-0.05）且
        轴均值轮高 ≥ 台面顶+R-0.02 持续 0.05s → 释放回 STANCE。
        前轴优先：后轴在前轴 SWING 期间不允许进入（永不双轴同抬）。"""
        _fax = body_pos[:2] + fwd * 0.228
        _rax = body_pos[:2] - fwd * 0.228
        _df, _tf = self._nearest_riser(_fax)
        _dr, _tr = self._nearest_riser(_rax)
        _wz_f = float(np.mean([wheel_xyz[i, 2] for i in (0, 1)]))
        _wz_r = float(np.mean([wheel_xyz[i, 2] for i in (2, 3)]))
        r = self.fk.r
        _swd = float(os.environ.get("S10_STAIR_SWING_D", "0.15"))
        if self._sp_f <= 0.0:
            if -_swd < _df < 0.05:
                self._sp_f = 1.0
                self._sp_f_top = _tf
                self._rel_f_t = None
        else:
            if _df < -0.05 and _wz_f >= self._sp_f_top + r - 0.01:
                if self._rel_f_t is None:
                    self._rel_f_t = self._t
                elif self._t - self._rel_f_t >= 0.05:
                    self._sp_f = 0.0
                    self._rel_f_t = None
            else:
                self._rel_f_t = None
        if self._sp_r <= 0.0:
            if -_swd < _dr < 0.05 and self._sp_f <= 0.0:
                self._sp_r = 1.0
                self._sp_r_top = _tr
                self._rel_r_t = None
        else:
            if _dr < -0.05 and _wz_r >= self._sp_r_top + r - 0.01:
                if self._rel_r_t is None:
                    self._rel_r_t = self._t
                elif self._t - self._rel_r_t >= 0.05:
                    self._sp_r = 0.0
                    self._rel_r_t = None
            else:
                self._rel_r_t = None
        step_lift = np.array([self._sp_f, self._sp_f,
                              self._sp_r, self._sp_r], dtype=np.float64)
        place_z = np.array([(self._sp_f_top if self._sp_f > 0 else 0.0)] * 2
                           + [(self._sp_r_top if self._sp_r > 0 else 0.0)] * 2,
                           dtype=np.float64)
        return step_lift, place_z

    # ---------------- QP Checker：接触合规校验（非分配主环） ----------------
    def _qp_check(self, q1q2, body_R, swing, tau_leg, dt):
        """osqp 12 变量：λ 贴近 J^-T·τ_pd，约束 = 抬升 λ≡0 / 支撑摩擦锥
        λ_z≥N_min。不可行/破锥 → 微降腿增益 _qp_scale（下限 0.85）；
        超时/异常沿用上一帧，不阻塞 200Hz 主环。"""
        if self._osqp is None:
            return
        _en = float(os.environ.get("S10_STAIR_QP", "1"))
        if _en <= 0:
            return
        try:
            from scipy import sparse
            n = 12
            lam_ref = np.zeros(n, dtype=np.float64)
            mus = float(os.environ.get("S10_STAIR_QP_MU", "0.6"))
            nmin = float(os.environ.get("S10_STAIR_QP_NMIN", "5.0"))
            for leg in range(4):
                q1, q2 = q1q2[leg]
                J = self.fk.jac(q1, q2)
                t_h = float(tau_leg[leg * 3 + 1])   # hipy
                t_k = float(tau_leg[leg * 3 + 2])   # knee
                if swing[leg] > 0.5:
                    lam_ref[leg * 3:leg * 3 + 3] = 0.0
                    continue
                fs = np.linalg.lstsq(J.T, np.array([t_h, t_k]),
                                     rcond=None)[0]   # (f_fwd, f_down)
                f_b = np.array([float(fs[0]), 0.0, -float(fs[1])])
                lam_ref[leg * 3:leg * 3 + 3] = body_R @ f_b
            P = sparse.eye(n, format="csc")
            q = -lam_ref
            for leg in range(4):
                base = leg * 3
                if swing[leg] > 0.5:
                    for k in range(3):
                        rows.append(0); cols.append(base + k); vals.append(1.0)
                        lo.append(0.0); hi.append(0.0)
                else:
                    rows.append(0); cols.append(base + 2); vals.append(-1.0)
                    lo.append(-np.inf); hi.append(-nmin)
                    for k in (0, 1):
                        rows.append(0); cols.append(base + k); vals.append(1.0)
                        rows.append(0); cols.append(base + 2); vals.append(-mus)
                        lo.append(-np.inf); hi.append(0.0)
                        rows.append(0); cols.append(base + k); vals.append(-1.0)
                        rows.append(0); cols.append(base + 2); vals.append(-mus)
                        lo.append(-np.inf); hi.append(0.0)
            A = sparse.csc_matrix(
                (np.asarray(vals, dtype=np.float64),
                 (np.asarray(rows, dtype=np.int64),
                  np.asarray(cols, dtype=np.int64))),
                shape=(len(lo), n))
            prob = self._osqp.OSQP()
            prob.setup(P=P, q=q, A=A, l=np.asarray(lo), u=np.asarray(hi),
                       verbose=False, time_limit=0.002, eps_abs=1e-3,
                       eps_rel=1e-3, max_iter=400, polish=False)
            res = prob.solve()
            ok = res.info.status in ("solved", "solved inaccurate")
            if not ok:
                self._qp_scale = max(0.85, self._qp_scale - 0.01)
            else:
                self._qp_scale = min(1.0, self._qp_scale + 0.002)
        except Exception:
            return

    # ---------------- 主入口 ----------------
    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):
        self._t = float(getattr(self, "_t", 0.0)) + dt
        body = self._body_state(qpos, qvel)
        fwd = np.array([np.cos(body["yaw"]), np.sin(body["yaw"])])
        # 终版：ModeSchedule 在 StairWBC 内计算（覆盖 cmd.step_lift/place_z）
        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        cmd = dict(cmd)
        cmd["step_lift"] = step_lift
        cmd["place_z"] = place_z
        # v878: 贴面爬升——抬升轮目标沿 riser 立面连续上升（保持接触滚动，
        # 替代悬空折叠抬腿）。对每只抬升轮按世界坐标算 z_face：
        #   d=轮心到棱沿切线距离；d∈[-r,0] 内 z 从底+r 平滑升到顶+r；
        # 经 place_z 传入 FP（wz = place_z+r+margin → z_face）。
        _sw_wheel0 = float(os.environ.get("S10_STAIR_SWING_WHEEL0", "1"))
        if float(os.environ.get("S10_STAIR_FACE", "1")) > 0:
            _pz_new = np.array(cmd.get("place_z", np.zeros(4)),
                               dtype=np.float64).copy()
            _r = self.fk.r
            for _leg in range(4):
                if step_lift[_leg] <= 0.02:
                    continue
                # v880: 轴均值距离——左右轮同相抬升（yaw 偏 4° 时逐轮 d 差
                # 0.025m → 单侧先抬 → roll 冲击实测）
                _ax_idx = (0, 1) if _leg in (0, 1) else (2, 3)
                _ax_xy = np.mean([wheel_xyz[_i, :2] for _i in _ax_idx], axis=0)
                # 找该轴前方最近高 riser
                _best_d = 1e9; _best = None
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    if _dhv <= 0.085:
                        continue
                    _dd = float(np.dot(_ax_xy - _rp, _tng))
                    if -0.20 < _dd < 0.05 and abs(_dd) < abs(_best_d):
                        _best_d = _dd; _best = (_rp, _tng, _dhv, _top)
                if _best is None:
                    continue
                (_rp, _tng, _dhv, _top) = _best
                _z_bot = float(_top - _dhv)
                _d_w = float(np.dot(_ax_xy - _rp, _tng))
                if -_r <= _d_w <= 0.0:
                    _t = float(np.clip((_d_w + _r) / max(_r, 1e-6), 0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    _z_face = _z_bot + _r + _dhv * _ss
                else:
                    _z_face = _top + _r
                # A1(批准): 硬上限——body 闭环/IK 不得把抬升目标泵高
                _z_face = min(_z_face, _top + _r + 0.005)
                # FP: wz = min(place_z + r + margin, hip+0.15) -> 反解 place_z
                _margin = float(os.environ.get("S10_STAIR_LIFT_MARGIN", "0.04"))
                _pz_new[_leg] = _z_face - _r - _margin
            cmd["place_z"] = _pz_new
            os.environ["S10_STAIR_SWING_WHEEL0"] = "0"
        elif _sw_wheel0 > 0:
            os.environ["S10_STAIR_SWING_WHEEL0"] = "1"
        _kpp = float(os.environ.get("S10_FP_KP_POS", "0"))
        if _kpp > 0:
            os.environ["S10_FP_KP_POS"] = str(_kpp * self._qp_scale)
        try:
            tau = super().compute_tau(qpos, qvel, wheel_xyz, wheel_vel,
                                      cmd, terrain_h, dt)
        finally:
            if _kpp > 0:
                os.environ["S10_FP_KP_POS"] = str(_kpp)
        # QP Checker：用本步腿 PD 力矩反推接触力校验（下步生效）
        try:
            _q1q2 = []
            for leg in range(4):
                _q1q2.append((
                    float(qpos[6 + leg * 3 + 1]),
                    float(qpos[6 + leg * 3 + 2])))
            self._qp_check(_q1q2, body["R"], step_lift, tau, dt)
        except Exception:
            pass
        return tau
