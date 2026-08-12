"""StairWBC：轮足狗楼梯爬升——位置基全身控制（终版 2026-08-11，清理版）。

架构（论文方向：位置基 + WBC 仅校验，几何相位硬切换）：
  ModeSchedule（整轴布尔相位，前轴→后轴，永不双轴同抬）
  → Body 姿态解析 + FootPlaceVMC 位置 PD（每腿 IK 放轮，静压）
  → WheelCtrl（支撑轮速度 PID 前驱 / 抬升轮 0）
  → QP Checker（osqp 接触合规校验，破锥仅微降腿增益，非力分配主环）
  → 腿控 Yaw（HipX 修正航向，轮差速在爬升期冻结）

清理说明（用户要求）：
- 所有参数收敛为类属性（不再散落 S10_STAIR_* / S10_QP_* env）；
- 删除门控：S10_STAIR_SWING_WHEEL0 状态翻转、S10_STAIR_QP/S10_STAIR_FACE
  开关（始终生效）；
- 修复 _qp_check 未初始化 rows/cols/vals/lo/hi 的 NameError（原被 except
  吞掉，QP Checker 从未真正运行）。
"""
import os

import numpy as np

from .stair_vmc_legs import (FootPlaceVMC, LEG_CTRL_IDX, LEG_QV_LEG,
                             WHEEL_Q_IDX, WHEEL_QV_IDX)


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
        # ---- 参数（终版固定值，收敛为类属性） ----
        self.swing_d = 0.15        # 抬升窗：与 ModeSchedule 触发窗对齐（v1008）
        # 0.30 提前预拉腿目标 → body 塌陷(0.63 实测)；0.15 保持 body 0.77
        self.swing_to = 1.5        # SWING 绝对超时兜底
        self.lift_margin = 0.04    # place_z -> 轮心目标 的余量
        self.qp_mu = 0.6           # QP Checker 摩擦系数
        self.qp_nmin = 5.0         # QP Checker 支撑轮法向下限
        self.qp_scale = 1.0        # 破锥时腿增益微降（下限 0.85）
        # ---- ModeSchedule 状态 ----
        self._sp_f = 0.0
        self._sp_r = 0.0
        self._sp_f_top = 0.0
        self._sp_r_top = 0.0
        self._rel_f_t = None
        self._rel_r_t = None
        self._sw_f_t0 = -1e9
        self._sw_r_t0 = -1e9
        self._t = 0.0
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
        """前/后轴布尔相位机：|d|<SWING_D 进入 SWING；过棱（d>0.10）且
        轴均值轮高 ≥ 台面顶+R+0.005 持续 0.05s → 释放回 STANCE。
        前轴优先：后轴在前轴 SWING 期间不允许进入（永不双轴同抬）。"""
        _fax = body_pos[:2] + fwd * 0.228
        _rax = body_pos[:2] - fwd * 0.228
        _df, _tf = self._nearest_riser(_fax)
        _dr, _tr = self._nearest_riser(_rax)
        _wz_f = float(np.mean([wheel_xyz[i, 2] for i in (0, 1)]))
        _wz_r = float(np.mean([wheel_xyz[i, 2] for i in (2, 3)]))
        r = self.fk.r
        if self._sp_f <= 0.0:
            if -self.swing_d < _df < 0.05 and self._sp_r <= 0.0:
                self._sp_f = 1.0
                self._sp_f_top = _tf
                self._rel_f_t = None
                self._sw_f_t0 = self._t
        else:
            if _df > 0.10 and _wz_f >= self._sp_f_top + r + 0.005:
                if self._rel_f_t is None:
                    self._rel_f_t = self._t
                elif self._t - self._rel_f_t >= 0.05:
                    self._sp_f = 0.0
                    self._rel_f_t = None
            else:
                self._rel_f_t = None
            if self._t - self._sw_f_t0 > self.swing_to:
                self._sp_f = 0.0
                self._rel_f_t = None
        if self._sp_r <= 0.0:
            if -self.swing_d < _dr < 0.05 and self._sp_f <= 0.0:
                self._sp_r = 1.0
                self._sp_r_top = _tr
                self._rel_r_t = None
                self._sw_r_t0 = self._t
        else:
            if _dr > 0.10 and _wz_r >= self._sp_r_top + r + 0.005:
                if self._rel_r_t is None:
                    self._rel_r_t = self._t
                elif self._t - self._rel_r_t >= 0.05:
                    self._sp_r = 0.0
                    self._rel_r_t = None
            else:
                self._rel_r_t = None
            if self._t - self._sw_r_t0 > self.swing_to:
                self._sp_r = 0.0
                self._rel_r_t = None
        step_lift = np.array([self._sp_f, self._sp_f,
                              self._sp_r, self._sp_r], dtype=np.float64)
        place_z = np.array([(self._sp_f_top if self._sp_f > 0 else 0.0)] * 2
                           + [(self._sp_r_top if self._sp_r > 0 else 0.0)] * 2,
                           dtype=np.float64)
        return step_lift, place_z

    # ---------------- 贴面爬升目标（几何，棱前提前抬） ----------------
    def _face_place_z(self, wheel_xyz, step_lift):
        """抬升轮目标沿 riser 立面连续上升（保持接触滚动，替代悬空折叠
        抬腿）。轴均值距离（左右同相），d∈[-SWING_D,0] 平滑 ramp 底+r→
        顶+r，d>0 台面顶+r；硬上限顶+r+0.005（防 body 闭环泵高）。"""
        _r = self.fk.r
        _pz = np.zeros(4)
        for _leg in range(4):
            if step_lift[_leg] <= 0.02:
                continue
            _ax_idx = (0, 1) if _leg in (0, 1) else (2, 3)
            _ax_xy = np.mean([wheel_xyz[_i, :2] for _i in _ax_idx], axis=0)
            _best_d = 1e9
            _best = None
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                if _dhv <= 0.085:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))
                if -self.swing_d < _dd < 0.05 and abs(_dd) < abs(_best_d):
                    _best_d = _dd
                    _best = (_rp, _tng, _dhv, _top)
            if _best is None:
                continue
            (_rp, _tng, _dhv, _top) = _best
            _z_bot = float(_top - _dhv)
            _d_w = float(np.dot(_ax_xy - _rp, _tng))
            if -self.swing_d <= _d_w <= 0.0:
                _t = float(np.clip((_d_w + self.swing_d) / self.swing_d,
                                   0.0, 1.0))
                _ss = _t * _t * (3.0 - 2.0 * _t)
                _z_face = _z_bot + _r + _dhv * _ss
            else:
                _z_face = _top + _r
            _z_face = min(_z_face, _top + _r + 0.005)
            _pz[_leg] = _z_face - _r - self.lift_margin
        return _pz

    # ---------------- QP Checker：接触合规校验（非分配主环） ----------------
    def _qp_check(self, q1q2, body_R, swing, tau_leg, dt):
        """osqp 12 变量：λ 贴近 J^-T·τ_pd，约束 = 抬升 λ≡0 / 支撑摩擦锥
        λ_z≥N_min。不可行/破锥 → 微降腿增益 qp_scale（下限 0.85）；
        超时/异常沿用上一帧，不阻塞 200Hz 主环。"""
        if self._osqp is None:
            return
        try:
            from scipy import sparse
            n = 12
            lam_ref = np.zeros(n, dtype=np.float64)
            for leg in range(4):
                q1, q2 = q1q2[leg]
                J = self.fk.jac(q1, q2)
                t_h = float(tau_leg[leg * 3 + 1])   # hipy
                t_k = float(tau_leg[leg * 3 + 2])   # knee
                if swing[leg] > 0.5:
                    continue
                fs = np.linalg.lstsq(J.T, np.array([t_h, t_k]),
                                     rcond=None)[0]   # (f_fwd, f_down)
                f_b = np.array([float(fs[0]), 0.0, -float(fs[1])])
                lam_ref[leg * 3:leg * 3 + 3] = body_R @ f_b
            P = sparse.eye(n, format="csc")
            q = -lam_ref
            rows, cols, vals, lo, hi = [], [], [], [], []
            for leg in range(4):
                base = leg * 3
                if swing[leg] > 0.5:
                    for k in range(3):
                        rows.append(len(lo)); cols.append(base + k)
                        vals.append(1.0); lo.append(0.0); hi.append(0.0)
                else:
                    rows.append(len(lo)); cols.append(base + 2)
                    vals.append(-1.0); lo.append(-np.inf)
                    hi.append(-self.qp_nmin)
                    for k in (0, 1):
                        rows.append(len(lo)); cols.append(base + k)
                        vals.append(1.0)
                        rows.append(len(lo)); cols.append(base + 2)
                        vals.append(-self.qp_mu)
                        lo.append(-np.inf); hi.append(0.0)
                        rows.append(len(lo)); cols.append(base + k)
                        vals.append(-1.0)
                        rows.append(len(lo)); cols.append(base + 2)
                        vals.append(-self.qp_mu)
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
                self.qp_scale = max(0.85, self.qp_scale - 0.01)
            else:
                self.qp_scale = min(1.0, self.qp_scale + 0.002)
        except Exception:
            return

    # ---------------- 闭式 2 连杆 IK（覆盖基类迭代式） ----------------
    def _ik(self, xd, zd, q1, q2, lift=False, leg=None):
        """闭式解，一次到位、不收敛到镜像折叠解。分支按腿固定：
        前腿(0,1) q2=+acos，后腿(2,3) q2=-acos（镜像腿）——按当前 q2
        符号选分支会卡在镜像折叠(一旦折叠 q2 变号→一直选错分支→永远
        折叠，FR 0.86/RR 0.81 实测)。lift 参数忽略（贴面爬升轮保持髋下）。"""
        L1, L2 = self.fk.L1, self.fk.L2
        zd_d = -zd   # 基类 zd 向上为正 -> 闭式用向下为正
        r2 = min(xd * xd + zd_d * zd_d, (L1 + L2) ** 2 - 1e-6)
        c2 = float(np.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2),
                           -1.0, 1.0))
        q2n = float(np.arccos(c2))
        if leg is not None and leg in (2, 3):
            q2n = -q2n
        elif leg is None and q2 < 0:
            q2n = -q2n
        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        # 用物理限位(±2.53/±2.72)——原 [-1.7,1.0] 挡住后腿标称 q1=+1.10
        # → 后腿达不到目标折叠上翻 0.98 实测。分支选择已防镜像，宽限安全。
        q1n = float(np.clip(q1n, -2.5, 2.5))
        q2n = float(np.clip(q2n, -2.7, 3.0))
        return q1n, q2n

    # ---------------- 几何地形（终版：世界坐标，不用 lidar） ----------------
    def _geo_terrain(self, wheel_xyz):
        """每轮支撑面高（世界坐标几何）：已过 riser → 最高已过顶；未过 →
        最近 riser 底（当前平台/地面）。替代 lidar terrain_h——棱口 lidar
        读高 0.7+ 会把支撑腿目标泵到 0.78+、body 抬到 1.0（首跑实测）。"""
        terr = []
        for leg in range(4):
            gt = 0.0
            best_d = 1e9
            best_bot = None
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                _dd = float(np.dot(wheel_xyz[leg, :2] - _rp, _tng))
                if _dd > 0.0:
                    gt = max(gt, float(_top))
                if abs(_dd) < abs(best_d):
                    best_d = _dd
                    best_bot = float(_top - _dhv)
            if gt > 0.4:
                terr.append(gt)
            elif best_bot is not None:
                terr.append(best_bot)
            else:
                terr.append(float(terrain_h[leg]))
        return np.asarray(terr, dtype=np.float64)

    # ---------------- 主入口 ----------------
    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):
        self._t = float(getattr(self, "_t", 0.0)) + dt
        body = self._body_state(qpos, qvel)
        fwd = np.array([np.cos(body["yaw"]), np.sin(body["yaw"])])
        # 终版：ModeSchedule + 贴面目标在 StairWBC 内计算（覆盖 cmd）
        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)
        self._dbg_phases = (self._sp_f, self._sp_r)
        self._dbg_phases = (self._sp_f, self._sp_r)
        cmd = dict(cmd)
        cmd["step_lift"] = step_lift
        cmd["place_z"] = place_z
        # 终版：支撑腿用几何地形（世界坐标），不用 lidar
        terrain_h = self._geo_terrain(wheel_xyz)
        # 后轴 SWING 期前腿加深静压（抗后轴抬升反作用，终版"后腿主动加
        # 垂直力抗抬头"对应项）——后轴抬轮把 body 后部顶起，前腿压载
        # 把 body 拉平，防前轮被反作用折叠上翻（0.86-0.91 实测）。
        _press_base = float(os.environ.get("S10_FP_PRESS", "0.005"))
        if self._sp_r > 0.5:
            os.environ["S10_FP_PRESS"] = str(float(os.environ.get(
                "S10_FP_PRESS_REAR", "0.030")))
        elif float(os.environ.get("S10_FP_PRESS", "0.005")) != _press_base:
            os.environ["S10_FP_PRESS"] = str(_press_base)
        # QP Checker 破锥时微降腿增益（经 S10_FP_KP_POS 传入基类）
        _kpp = float(os.environ.get("S10_FP_KP_POS", "0"))
        if _kpp > 0:
            os.environ["S10_FP_KP_POS"] = str(_kpp * self.qp_scale)
        try:
            tau = super().compute_tau(qpos, qvel, wheel_xyz, wheel_vel,
                                      cmd, terrain_h, dt)
        finally:
            if _kpp > 0:
                os.environ["S10_FP_KP_POS"] = str(_kpp)
        # 爬升瞬态：轮矩纯前驱（冻结差速，航交由 HipX 修正）。
        # 前驱下限：狗撞棱 body 停、后轮空转超速被 PID 倒转(tauW=+9/10
        # 实测) → 狗被夹死。SWING 期支撑轮至少 -DRIVE_FLOOR 前驱。
        _df = -6.0
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        try:
            _vx_f = float(getattr(self, "_vx_f", 0.0))
            _any_sw = float(np.max(step_lift)) > 0.5
            for _leg in range(4):
                _wq = float(qvel[WHEEL_QV_IDX[_leg]])
                _vw = -_wq * self.fk.r
                _vref = _vx_f
                if _any_sw:
                    # 前轮抬空后失去 yaw 阻力，支撑轮差速主动抗旋（yaw_rate
                    # 反馈）；同时保持前驱下限防后轮空转倒转
                    _vref = _vx_f - _side_s[_leg] * float(qvel[5]) * 2.0                         * self.track_half
                _tw = (-(self.wheel_k * (_vref - _vw))
                       - self.wheel_d * _wq)
                if _any_sw:
                    _tw = max(float(_tw), _df)
                tau[WHEEL_Q_IDX[_leg]] = float(np.clip(_tw, -13.5, 13.5))
        except Exception:
            pass
        # 阶段 A：位形恢复（终版批准）——前腿 relx<0（后折）或垂距<3cm
        # （近水平）时，关节空间直接加恢复矩（不经过 J^T，奇异区不被
        # 吃掉），把腿掰回前伸位形；同时后轮满驱推身、前轮微正转。
        # 退出条件：relx>0 且垂距>5cm 持续 0.1s（阶段 B 几何举身接管）。
        try:
            _recov_k = float(os.environ.get("S10_FP_RECOV_K", "15.0"))
            _recov_on = False
            for _leg in (0, 1):
                _hip_r = body["pos"] + body["R"] @ np.array(
                    [0.2277, 0.0, 0.0])
                _relx_r = (np.cos(body["yaw"])
                           * (wheel_xyz[_leg, 0] - _hip_r[0])
                           + np.sin(body["yaw"])
                           * (wheel_xyz[_leg, 1] - _hip_r[1]))
                _drop_r = float(_hip_r[2] - wheel_xyz[_leg, 2])
                # v1011: 仅当前腿未 SWING 时触发——SWING 期轮在髋附近
                # (垂距<3cm 正常)，恢复矩误触发 + 后轮前驱 → 弹射
                if (step_lift[_leg] <= 0.5
                        and (_relx_r < 0.0 or _drop_r < 0.03)):
                    _recov_on = True
                    _q1 = float(qpos[6 + _leg * 3 + 1])
                    _tau_r = _recov_k * max(0.05 - _relx_r, 0.0)
                    # 终版公式：K_recov=15 Nm/m * relx 误差，温和恢复
                    _tau_r = float(np.clip(_tau_r, -48.0, 48.0))
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r
            if _recov_on:
                _rdrive = -float(os.environ.get("S10_FP_RECOV_DRIVE", "8.0"))
                for _leg in (2, 3):
                    tau[WHEEL_Q_IDX[_leg]] = _rdrive
                for _leg in (0, 1):
                    if float(step_lift[_leg]) <= 0.5:
                        tau[WHEEL_Q_IDX[_leg]] = -3.0
        except Exception:
            pass
        # v1021/v1023: 支撑轮前驱下限 + yaw_rate 差速——爬升期后轮空转被
        # PID 倒转 → 卡死；且前轮贴面不对称让 yaw 漂到 2.1(偏 34°)，前驱
        # 全变西向推力(实测)。SWING 期支撑轮至少 -DRIVE_FLOOR 前驱(负=前)
        # + yaw_rate 差速把航向拉回。
        try:
            _any_swx = float(np.max(step_lift)) > 0.5
            if _any_swx:
                _dfx = -float(os.environ.get("S10_FP_DRIVE_FLOOR", "6.0"))
                _kd_yx = float(os.environ.get("S10_FP_YAW_DIFF", "2.0"))
                for _leg in range(4):
                    if step_lift[_leg] <= 0.5:
                        _sx = -1.0 if _leg in (0, 1) else 1.0
                        _corr = _sx * float(qvel[5]) * _kd_yx * self.track_half
                        tau[WHEEL_Q_IDX[_leg]] = min(
                            float(tau[WHEEL_Q_IDX[_leg]]), _dfx) - _corr
        except Exception:
            pass
        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）
        try:
            _kp_y = float(os.environ.get("S10_FP_YAW_KP", "2.0"))
            _kd_y = float(os.environ.get("S10_FP_YAW_KD", "0.5"))
            _yerr = 0.0
            if self.stair is not None:
                _yerr = float(getattr(self.stair, "_last_err", 0.0))
            _yr = float(body["omega"]) if "omega" in body else float(qvel[5])
            _th_y = _kp_y * _yerr - _kd_y * _yr
            tau[LEG_CTRL_IDX[0]] += _th_y     # FL hipx
            tau[LEG_CTRL_IDX[3]] -= _th_y     # FR hipx
            tau[LEG_CTRL_IDX[6]] += 0.5 * _th_y   # RL hipx
            tau[LEG_CTRL_IDX[9]] -= 0.5 * _th_y   # RR hipx
        except Exception:
            pass
        # 最终力矩限幅（防恢复矩/防过伸叠加超限）
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -48, 48)
        tau[WHEEL_Q_IDX] = np.clip(tau[WHEEL_Q_IDX], -13.5, 13.5)
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