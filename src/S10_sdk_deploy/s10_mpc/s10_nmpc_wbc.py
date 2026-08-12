"""s10_nmpc_wbc.py — NMPC+WBC 轮足爬梯执行层（论文方案骨架，2026-08-12 收口后启动）。

三层架构（替代位置基 StairWBC 的 0.125m 爬升执行）：
- 轨迹层：stair_world 几何 → body 参考（vx/z/pitch/heading）随台阶推进。
- NMPC（20Hz）：SRBD（单刚体）接触力优化——m·a=ΣF+mg、I·α=Σr×F，
  摩擦锥 + 抬升轮 F=0（ModeSequence），输出期望接触力 F_des 与 body 加速度。
- WBC（200Hz）：支撑腿 J^T·F_des 力分配 + 摆腿位置 PD + 轮 Pfaffian 驱动。

纯 numpy + osqp，接口与 stair_vmc_noros.py 的 compute_tau 一致。
"""
import os
import numpy as np

LEG_ATTACH = np.array([
    [0.2277, 0.181191], [-0.2277, 0.181191],  # fl, rl (body y+)
    [0.2277, -0.181191], [-0.2277, -0.181191],  # fr, rr
], dtype=np.float64)
LEG_Q_IDX = [7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21]
LEG_CTRL_IDX = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
LEG_QV_LEG = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
WHEEL_Q_IDX = [3, 7, 11, 15]
WHEEL_QV_IDX = [9, 13, 17, 21]
WHEEL_BODY = [5, 9, 13, 17]


class S10LegFK:
    def __init__(self, L1=0.18, L2=0.18, r=0.081):
        self.L1, self.L2, self.r = L1, L2, r

    def wheel_pos(self, q1, q2):
        px = self.L1 * np.sin(q1) + self.L2 * np.sin(q1 + q2)
        pz = self.L1 * np.cos(q1) + self.L2 * np.cos(q1 + q2)
        return np.array([px, pz])

    def jac(self, q1, q2):
        c1, s1 = np.cos(q1), np.sin(q1)
        c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
        return np.array([
            [self.L1 * c1 + self.L2 * c12, self.L2 * c12],
            [-self.L1 * s1 - self.L2 * s12, -self.L2 * s12],
        ])


class NmpcWbc:
    """NMPC+WBC 轮足爬梯控制器（S10 机体）。"""

    def __init__(self, mass=19.0, g=9.81, L1=0.18, L2=0.18, r=0.081,
                 track_half=0.24):
        self.fk = S10LegFK(L1, L2, r)
        self.m, self.g = mass, g
        self.track_half = track_half
        self.wheelbase = 0.456
        # 总惯量（近似：base 8.02kg diag + 四腿，绕 CoM）
        self.I_body = np.diag([0.15, 0.22, 0.30])
        self.stair = None
        self.stair_world = []
        self.swing_d = float(os.environ.get('S10_NMPC_SWING_D', '0.35'))
        self.swing_to = 1.2
        self.mu = float(os.environ.get('S10_NMPC_MU', '0.8'))
        self.fz_max = float(os.environ.get('S10_NMPC_FZ_MAX', '180.0'))
        self.nmpc_hz = float(os.environ.get('S10_NMPC_HZ', '20.0'))
        self.nmpc_horizon = int(os.environ.get('S10_NMPC_H', '4'))
        # 轨迹层状态
        self._vx_f = 0.0
        self._om_f = 0.0
        self._t = 0.0
        self._sp_f = 0.0
        self._sp_r = 0.0
        self._sp_f_hover = 0.0
        self._sp_f_hover_s = 0.0
        self._sp_f_hover_t0 = -1e9
        self._sp_f_hover_pos = None
        self._sp_f_win_t0 = -1e9
        self._sp_f_top = 0.0
        self._sp_r_top = 0.0
        self._sw_f_t0 = -1e9
        self._sw_r_t0 = -1e9
        self._nmpc_prob = None
        try:
            import osqp
            self._osqp = osqp
        except Exception:
            self._osqp = None
        self._dbg_phases = (0.0, 0.0)
        self._dbg_fdes = np.zeros(12)

    # ---------------- 身体状态 ----------------
    def _body_state(self, qpos, qvel):
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
        vw = R.T @ np.asarray(qvel[0:3], dtype=np.float64)
        return dict(pos=qpos[0:3], yaw=yaw, roll=roll, pitch=pitch,
                    vx=float(vw[0]), R=R,
                    omega=np.asarray(qvel[3:6], dtype=np.float64))

    # ---------------- ModeSequence：抬升轮（F=0）----------------
    def _nearest_riser(self, ax):
        dmin, top = 1e9, 0.0
        for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
            if _dhv <= 0.085:      # riser1 纯滚（与 v1050 一致）
                continue
            _dd = float(np.dot(np.asarray(ax, dtype=np.float64) - _rp, _tng))
            if abs(_dd) < abs(dmin):
                dmin, top = _dd, float(_top)
        return dmin, top

    def _mode_sequence(self, body_pos, fwd, wheel_xyz):
        """布尔接触模式：前/后轴进入抬升窗 → 该轴轮 F=0（SWING）。
        轴内左右同步（v1040 结论），过棱且轮高≥台面顶+半径-0.01 释放。"""
        _fax = body_pos[:2] + fwd * 0.228
        _rax = body_pos[:2] - fwd * 0.228
        _perp = np.array([-fwd[1], fwd[0]])
        # v1117: ??????????yaw ????????????????
        # ?????????????????????? yaw ???real43/47
        # riser2 ?? SWING ? om 1.5-3 ????
        _dfl, _tfl = self._nearest_riser(_fax + _perp * 0.181)
        _dfr, _tfr = self._nearest_riser(_fax - _perp * 0.181)
        if _dfl <= _dfr:
            _df, _tf = _dfl, _tfl
        else:
            _df, _tf = _dfr, _tfr
        _drl, _trl = self._nearest_riser(_rax + _perp * 0.181)
        _drr, _trr = self._nearest_riser(_rax - _perp * 0.181)
        if _drl <= _drr:
            _dr, _tr = _drl, _trl
        else:
            _dr, _tr = _drr, _trr
        _wz_f = float(np.mean([wheel_xyz[i, 2] for i in (0, 1)]))
        _wz_r = float(np.mean([wheel_xyz[i, 2] for i in (2, 3)]))
        r = self.fk.r
        if self._sp_f <= 0.0:
            if not (-self.swing_d < _df < 0.05):
                self._sp_f_win_t0 = -1e9
            if -self.swing_d < _df < 0.05 and self._sp_r <= 0.0:
                # v1110: ??????????? SWING ??????????
                # ????? z >= ???-0.125+??-0.02?????????
                # ?????????????????????real42 ????
                # ?? riser1 ??y ? 38.0?28s ????????
                if _wz_r >= _tf - 0.125 + r - 0.02:
                    self._sp_f = 1.0
                    self._sp_f_top = _tf
                    self._sw_f_t0 = self._t
                else:
                    # v1155: ???????????????????????
                    # ???real89 ??? 8s???????? 2s ?? SWING
                    # ???????????????
                    if self._sp_f_win_t0 < 0:
                        self._sp_f_win_t0 = self._t
                    if self._t - self._sp_f_win_t0 > 2.0:
                        self._sp_f = 1.0
                        self._sp_f_top = _tf
                        self._sw_f_t0 = self._t
        else:
            # v1079(方向1): 前轮过棱后进 HOVER——保持正 drop（body 相对
            # 目标，非台面+R）随 body 平移，平移 hover_len 后 STANCE。
            # world 目标在 body 抬升时 drop 收缩→折叠；body 相对目标
            # drop 恒定→腿保持弯曲不折叠，且前轮不空转自旋。
            if _wz_f >= self._sp_f_top + r - 0.01:
                if self._sp_f_hover <= 0.5:
                    self._sp_f_hover = 1.0
                    self._sp_f_hover_s = float(getattr(
                        self.stair, '_s_cur', 0.0))
                    self._sp_f_hover_t0 = self._t
                    self._sp_f_hover_pos = np.asarray(
                        body_pos[:2], dtype=np.float64)
                # v1082: ???????????? / ??????? / ?? /
                # ?????????????? _s_cur ????????
                _h_len = float(os.environ.get('S10_NMPC_HOVER_LEN', '0.10'))
                _h_tmax = float(os.environ.get('S10_NMPC_HOVER_TMAX', '0.5'))
                _h_done = False
                if float(getattr(self.stair, '_s_cur', 0.0)) - \
                        self._sp_f_hover_s >= _h_len:
                    _h_done = True
                if not _h_done and self._sp_f_hover_pos is not None:
                    _dp = np.asarray(body_pos[:2], dtype=np.float64) - \
                        self._sp_f_hover_pos
                    if float(_dp[0] * fwd[0] + _dp[1] * fwd[1]) >= _h_len:
                        _h_done = True
                if not _h_done and \
                        self._t - self._sp_f_hover_t0 >= _h_tmax:
                    _h_done = True
                if not _h_done and \
                        _wz_f <= self._sp_f_top + r - 0.04:
                    _h_done = True
                if _h_done:
                    self._sp_f = 0.0
                    self._sp_f_hover = 0.0
                    if os.environ.get('S10_NMPC_DEBUG', '0') == '1':
                        print('[HOVER] exit t=%.2f y=%.2f wz=%.3f' % (
                            self._t, float(body_pos[1]), _wz_f), flush=True)
            elif self._t - self._sw_f_t0 > self.swing_to:
                self._sp_f = 0.0
                self._sp_f_hover = 0.0
        if self._sp_r <= 0.0:
            if -self.swing_d < _dr < 0.05 and (
                    self._sp_f <= 0.0 or self._sp_f_hover > 0.5):
                self._sp_r = 1.0
                self._sp_r_top = _tr
                self._sw_r_t0 = self._t
        else:
            if _wz_r >= self._sp_r_top + r - 0.01:
                self._sp_r = 0.0
            elif self._t - self._sw_r_t0 > self.swing_to:
                self._sp_r = 0.0
        swing = np.array([self._sp_f, self._sp_f,
                          self._sp_r, self._sp_r], dtype=np.float64)
        self._dbg_phases = (self._sp_f, self._sp_r)
        return swing

    # ---------------- 轨迹层：body 参考 ----------------
    def _ref_traj(self, body, vx_cmd, om_cmd, terrain_h):
        """台阶推进参考：vx 插值、body z 跟轮下地形均值+偏移、pitch 随坡度、
        heading 锁楼梯切线（v1034 思路）。"""
        r = self.fk.r
        _z_geo = []
        for leg in range(4):
            gt = 0.0
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                _dd = float(np.dot(
                    np.asarray(body['pos'][:2], dtype=np.float64)
                    - _rp, _tng))
                # v1153: z ???? 0.4m?????? riser?_dd>0??????
                # ????? SWING ???????79 ???????????
                # ???? SWING ??????????????
                if _dd > -0.4:
                    gt = max(gt, float(_top))
            _z_geo.append((gt if gt > 0.4 else float(terrain_h[leg])) + r)
        z_ref = float(np.mean(_z_geo)) + float(os.environ.get(
            'S10_NMPC_Z_OFF', '0.20'))
        _pitch_ref = -float(np.arctan2(
            float(np.mean(_z_geo[0:2])) - float(np.mean(_z_geo[2:4])),
            0.456))
        # 楼梯切线 heading
        hdg = float(np.arctan2(self.stair_world[0][1][1],
                               self.stair_world[0][1][0])) \
            if self.stair_world else float(body['yaw'])
        return dict(vx=vx_cmd, z=z_ref, pitch=_pitch_ref, hdg=hdg)

    # ---------------- NMPC：SRBD 接触力优化（20Hz）----------------
    def _nmpc(self, body, ref, swing, wheel_xyz, dt_nmpc):
        """SRBD QP：变量 [F(12), a(3)]，等式 m·a=ΣF+mg，
        代价跟踪 a_des 与角动量 M_des=Σr×F，摩擦锥/法向/抬升轮 F=0。
        osqp 问题固定结构，每解只 update 数据（<2ms）。"""
        m, g = self.m, self.g
        R = body['R']
        kp_z = float(os.environ.get('S10_NMPC_KP_Z', '200.0'))
        kd_z = float(os.environ.get('S10_NMPC_KD_Z', '30.0'))
        kp_vx = float(os.environ.get('S10_NMPC_KP_VX', '10.0'))
        kp_p = float(os.environ.get('S10_NMPC_KP_PITCH', '300.0'))
        kd_p = float(os.environ.get('S10_NMPC_KD_PITCH', '30.0'))
        kp_y = float(os.environ.get('S10_NMPC_KP_YAW', '2.0'))
        kd_y = float(os.environ.get('S10_NMPC_KD_YAW', '2.0'))
        w_f = float(os.environ.get('S10_NMPC_WF', '1e-3'))
        w_a = float(os.environ.get('S10_NMPC_WA', '1.0'))
        w_m = float(os.environ.get('S10_NMPC_WM', '0.1'))
        # 名义支撑力参考：防 QP 把 F 压到 0 → 自由落体（台架 F=0 实测）
        w_fr = float(os.environ.get('S10_NMPC_WFR', '0.02'))
        # v1060/v1061: 贴面爬升——SWING 轮保留小接触力（F_z≥5N）；
        # F_ref 按实际接触轮数分配（前轮抬升时后轮各担 mg/2，否则
        # 求解器给后轮 0 → body 落 → 前轮饱和 → 振荡发射实测）
        F_ref = np.zeros(12)
        _n_cont = max(float(np.sum(swing <= 0.5)), 1.0)
        for i in range(4):
            F_ref[3*i+2] = m * g / _n_cont * (
                1.0 if swing[i] <= 0.5 else 0.2)
        # 期望线性加速度（世界系）
        a_des = np.zeros(3)
        a_des[2] = kp_z * (ref['z'] - body['pos'][2]) \
            - kd_z * float(np.dot(R[:, 2], body['omega']))
        fwd_w = R @ np.array([1.0, 0.0, 0.0])
        v_w = fwd_w * body['vx']
        a_des[0:2] = kp_vx * (ref['vx'] * fwd_w[0:2] - v_w[0:2])
        # 期望角加速度（pitch/yaw，roll 回零）
        al_des = np.zeros(3)
        al_des[1] = kp_p * (ref['pitch'] - body['pitch']) \
            - kd_p * float(np.dot(R[:, 1], body['omega']))
        # v1072: 轴抬升 pitch 前馈——后轴 SWING 减载反作用使 body 翘头
        # （~21Nm），前腿强保持饱和仍上折；前馈主动低头抵消（F_z_min=46
        # 后后轮能出力执行）
        _ff_sw = float(os.environ.get('S10_NMPC_PITCH_FF_SW', '0.0'))
        if _ff_sw > 0.0:
            if float(np.max(swing[2:4])) > 0.5:
                al_des[1] -= _ff_sw
            elif float(np.max(swing[0:2])) > 0.5:
                al_des[1] += _ff_sw
        al_des[2] = kp_y * (ref['hdg'] - body['yaw']) \
            - kd_y * float(np.dot(R[:, 2], body['omega']))
        al_des[0] = -3.0 * body['roll']
        # v1082: SWING/HOVER ??? z/????????a_des[2]<-g ?????
        # ? F_z>=0 ?? ? ?????? F=0 ?????v1081 HOVER ???
        # F ????????????????F ?????
        a_des[2] = float(np.clip(a_des[2], float(os.environ.get(
            'S10_NMPC_AZ_MIN', '-4.0')), float(os.environ.get(
                'S10_NMPC_AZ_MAX', '12.0'))))
        if float(np.max(swing)) > 0.5:
            _al_lim = float(os.environ.get('S10_NMPC_AL_LIM', '6.0'))
            al_des[1] = float(np.clip(al_des[1], -_al_lim, _al_lim))
            # v1147: SWING ? pitch ?????pitch<-0.55???>32?????
            # ???????0.36m?????? ? ???????real83 ??
            # 1.35 ????????????????????????
            if float(body['pitch']) < -0.55:
                al_des[1] += 20.0 * (-0.55 - float(body['pitch']))
            # v1083/v1113: SWING ? yaw ???????real13/46 ?????
            # ????+???????????+hipx ?????
            al_des[2] = 0.0
            _fwd2 = fwd_w[0:2]
            _n2 = float(np.dot(_fwd2, _fwd2))
            if _n2 > 1e-6:
                a_des[0:2] = (float(np.dot(a_des[0:2], _fwd2)) / _n2) * _fwd2
        M_des = self.I_body @ al_des \
            + np.cross(body['omega'], self.I_body @ body['omega'])
        # 轮相对 CoM 位置（世界系）
        r_w = wheel_xyz - body['pos']
        # A_m：F(12) -> Σ r×F (3)
        A_m = np.zeros((3, 12))
        for i in range(4):
            rc = r_w[i]
            A_m[0, 3*i+1] = -rc[2]
            A_m[0, 3*i+2] = rc[1]
            A_m[1, 3*i+0] = rc[2]
            A_m[1, 3*i+2] = -rc[0]
            A_m[2, 3*i+0] = -rc[1]
            A_m[2, 3*i+1] = rc[0]
        # 代价矩阵 P（15x15）——QP 标准形 0.5·x^T P x，P=2·(w‖·‖²)，
        # 否则线性项被放大 → QP 选 F=0 自由落体（台架实测）
        n = 15
        P = np.zeros((n, n))
        P[0:12, 0:12] = 2.0 * (
            w_f * np.eye(12) + w_m * (A_m.T @ A_m) + w_fr * np.eye(12))
        P[12:15, 12:15] = 2.0 * w_a * np.eye(3)
        q = np.zeros(n)
        q[12:15] = -2.0 * w_a * a_des
        q[0:12] = -2.0 * (w_m * (A_m.T @ M_des) + w_fr * F_ref)
        # 等式：m·a - ΣF = mg
        Ae = np.zeros((3, n))
        Ae[0, 0:12:3] = -1.0
        Ae[1, 1:12:3] = -1.0
        Ae[2, 2:12:3] = -1.0
        Ae[0, 12] = m
        Ae[1, 13] = m
        Ae[2, 14] = m
        # m·a - ΣF = m·g_vec，g_vec=(0,0,-g)（重力向下）——写 +mg 会让
        # ΣF 解为向下、WBC 把 body 往上推发射（台架翻车实测）
        # v1087(??3????): ????? mode ????SWING ????
        # ?/???WBC ????? PD???? F_des?????????
        # SWING ? F_z ? ????????????? ? ???????
        # ?????real14-18 ?? bz 0.70?fn ????
        # v1109: ??????/?? SWING ??"????"?????????
        # ?? F_z>=20 ??????????? F_z>=46?v1070??
        be = np.array([0.0, 0.0, -m * g])
        # 不等式（固定结构）：每轮 6 行
        rows = []
        ub = []
        for i in range(4):
            e = np.zeros(n)
            e[3*i+2] = -1.0
            rows.append(e); ub.append(0.0)
            for j in range(2):
                e = np.zeros(n)
                e[3*i+j] = 1.0; e[3*i+2] = -self.mu
                rows.append(e); ub.append(0.0)
                e = np.zeros(n)
                e[3*i+j] = -1.0; e[3*i+2] = -self.mu
                rows.append(e); ub.append(0.0)
            e = np.zeros(n)
            e[3*i+2] = 1.0
            rows.append(e); ub.append(self.fz_max)
        # 恒等行把 F 变量界折进 A（结构固定，l/u 随调用变化）：
        # 抬升轮 F=0（l=u=0），支撑轮自由（l=-inf,u=+inf）
        A = np.vstack([Ae] + rows + [np.eye(n)[0:12]])
        l = np.hstack([be] + [-1e9] * len(rows)
                      + [-1e9] * 12)
        u = np.hstack([be] + ub + [1e9] * 12)
        for i in range(4):
            if swing[i] > 0.5:
                # v1081: 前轴 SWING F=0（滚动越阶，v1080 实测前轮能滚过
                # riser2 y=38.3）、后轴 SWING F_z≥46（滚爬，v1070 实测后轮
                # 需要力才能爬）——不对称接触界
                # v1109: ?? SWING F_z>=20??????????? F_z>=46
                _fz_min_sw = float(os.environ.get(
                    'S10_NMPC_SWING_FZ_MIN', '46.0'))
                if i in (0, 1):
                    _fz_min_sw = float(os.environ.get(
                        'S10_NMPC_FRONT_SWING_FZ_MIN', '20.0'))
                l[3 + len(rows) + 3*i + 2] = _fz_min_sw
                u[3 + len(rows) + 3*i + 2] = self.fz_max
            elif float(np.max(swing[0:2])) > 0.5 and i in (2, 3):
                # v1159: ?? SWING ???????F_z >= ??? 95N??85 ?
                # ?????????????????????? 124N<186N??
                l[3 + len(rows) + 3*i + 2] = float(os.environ.get(
                    'S10_NMPC_STANCE_FZ_MIN', '95.0'))
                u[3 + len(rows) + 3*i + 2] = self.fz_max
            elif float(np.max(swing[2:4])) > 0.5 and i in (0, 1):
                # v1159: ?? SWING ????????? F_z >= 95N?
                l[3 + len(rows) + 3*i + 2] = float(os.environ.get(
                    'S10_NMPC_STANCE_FZ_MIN', '95.0'))
                u[3 + len(rows) + 3*i + 2] = self.fz_max
        import osqp
        from scipy import sparse
        if self._nmpc_prob is None:
            prob = osqp.OSQP()
            prob.setup(P=sparse.csc_matrix(P), q=q,
                       A=sparse.csc_matrix(A), l=l, u=u,
                       verbose=False, eps_abs=1e-3, eps_rel=1e-3,
                       max_iter=500, polish=False)
            self._nmpc_prob = prob
        else:
            prob = self._nmpc_prob
            prob.update(P=sparse.csc_matrix(P), q=q,
                        A=sparse.csc_matrix(A), l=l, u=u)
        try:
            import time as _t
            _t0 = _t.perf_counter()
            res = prob.solve()
            x = np.asarray(res.x).reshape(n)
            F = x[0:12].reshape(4, 3)
            a = x[12:15]
            if os.environ.get('S10_NMPC_DEBUG', '0') == '1':
                print('[NMPC] t=%.2f F=%s a=%s ades=%s Mdes=%s st=%s dt=%.1fms'
                      % (self._t, np.round(F, 1).tolist(),
                         np.round(a, 2).tolist(),
                         np.round(a_des, 2).tolist(),
                         np.round(M_des, 1).tolist(),
                         res.info.status,
                         1e3 * (_t.perf_counter() - _t0)), flush=True)
            return F, a
        except Exception as _e:
            print('[NMPC] ERR', _e, flush=True)
            return np.zeros((4, 3)), a_des

    # ---------------- WBC：腿力分配 + 摆腿 PD + 轮驱动 ----------------
    def _wbc(self, qpos, qvel, wheel_xyz, body, F_des, swing, cmd,
             terrain_h, ref, dt):
        tau = np.zeros(16, dtype=np.float64)
        R = body['R']
        r = self.fk.r
        # 摆腿目标（贴面弧：z = z_bot + sqrt(R²-d²)，v1050 后仅 riser2+）
        swing_tgt_z = {}
        for leg in range(4):
            sl = swing[leg]
            if sl > 0.5:
                _ax_idx = (0, 1) if leg in (0, 1) else (2, 3)
                _ax_xy = np.mean([wheel_xyz[_i, :2] for _i in _ax_idx], axis=0)
                _best_d, _best = 1e9, None
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    if _dhv <= 0.085:
                        continue
                    _dd = float(np.dot(_ax_xy - _rp, _tng))
                    if -self.swing_d < _dd < 0.05 and abs(_dd) < abs(_best_d):
                        _best_d, _best = _dd, (_rp, _tng, _dhv, _top)
                if _best is not None:
                    (_rp, _tng, _dhv, _top) = _best
                    _z_bot = float(_top - _dhv)
                    _d_w = _best_d
                    # v1058: 爬升斜坡（原 sqrt(R²-d²) 是下降滚动弧，目标低于
                    # 地面 0.55 → 轮目标乱摆发射）；正确：地面+r → 台面顶+r
                    # 随窗 smoothstep（位置基 _face_place_z 同款，已验证）
                    # v1163: ??? 0.7??v1159 ?? 0.86 ????????0.15?
                    # ????????? riser3 ?????real95 ? 1.11??
                    # ???????????v1157 ???????????????
                    _t = float(np.clip(
                        (_d_w + self.swing_d) / (0.7 * self.swing_d),
                        0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    _zc = _z_bot + r + _dhv * _ss
                    swing_tgt_z[leg] = min(_zc, _top + r + 0.005)
        # 每腿力矩
        for leg in range(4):
            b = leg * 3
            qhx = float(qpos[LEG_Q_IDX[b]])
            q1 = float(qpos[LEG_Q_IDX[b + 1]])
            q2 = float(qpos[LEG_Q_IDX[b + 2]])
            dq1 = float(qvel[6 + LEG_QV_LEG[b + 1]])
            dq2 = float(qvel[6 + LEG_QV_LEG[b + 2]])
            hip_w = body['pos'] + R @ np.array(
                [LEG_ATTACH[leg][0], LEG_ATTACH[leg][1], 0.0])
            sl = swing[leg]
            if sl > 0.5 and leg in swing_tgt_z:
                # 摆腿（贴面爬升）：位置 PD + 小支撑力（F_des 的 F_z 部分）
                wz_t = swing_tgt_z[leg]
                # v1079(方向1): HOVER 期前轮目标 = body 相对 drop（正 drop
                # 恒定，轮随 body 平移），钳在台面顶+半径以上不插台面
                if self._sp_f_hover > 0.5 and leg in (0, 1):
                    _hd = float(os.environ.get('S10_NMPC_HOVER_DROP', '0.03'))
                    wz_t = float(hip_w[2]) - _hd
                    wz_t = min(wz_t, self._sp_f_top + r + 0.008)
                    wz_t = max(wz_t, self._sp_f_top + r - 0.005)
                # v1089: ???????????????????-2cm?????
                # ???+R????????????????????????
                # J^T ??????????? body ???real19 ?? body 0.63?
                # ? 0.75 ????????????? z ???????????
                # v1146: ??????v1085b ?????????????-2cm?
                # ?? SWING ? body ??????? 0.64??? 0.747 ???
                # ? ????????real78 ?? 0.91 ???????????
                # ??????? J^T ???????
                wz_t = min(wz_t, float(hip_w[2]) - 0.02)

                rel = np.array([
                    wheel_xyz[leg, 0] - hip_w[0],
                    wheel_xyz[leg, 1] - hip_w[1],
                    wz_t - hip_w[2]])
                cy, sy = np.cos(body['yaw']), np.sin(body['yaw'])
                relb = np.array([cy * rel[0] + sy * rel[1],
                                 -sy * rel[0] + cy * rel[1],
                                 rel[2]])
                relb[0] = max(float(relb[0]), 0.0)
                _lo = float(os.environ.get('S10_NMPC_REACH', '-0.34'))
                _rz = float(np.clip(relb[2], _lo, -0.02))
                q1t, q2t = self._ik(float(relb[0]), _rz, q1, q2, leg=leg)
                # v1075: 后轴 SWING 低增益（少抬多滚）——前轮已证明贴面滚爬
                # 能越阶；后轴强位置引导会把 body 顶起俯仰 → 前腿上折。
                # 后轴 KP 40 让轮主要靠 F_z≥46 滚上立面，俯仰小
                _kp_sw = float(os.environ.get('S10_NMPC_KP_SW', '120.0'))
                if leg in (2, 3):
                    _kp_sw = float(os.environ.get('S10_NMPC_KP_SW_R', '40.0'))
                kp = _kp_sw
                kd = float(os.environ.get('S10_NMPC_KD_SW', '6.0'))
                tau[LEG_CTRL_IDX[b + 1]] = kp * (q1t - q1) - kd * dq1
                tau[LEG_CTRL_IDX[b + 2]] = kp * (q2t - q2) - kd * dq2
                # v1067: 过伸回压——轮高于目标时经 J^T 直接下压
                # （位置基 v1017 同款；否则轮悬 1.1 泵高发射实测）
                _dz_ov = float(wheel_xyz[leg, 2] - wz_t - 0.02)
                if _dz_ov > 0.0:
                    _fs_ov = np.array([0.0, _dz_ov * 300.0])
                    _J = self.fk.jac(q1, q2)
                    _th1o, _th2o = _J.T @ _fs_ov
                    tau[LEG_CTRL_IDX[b + 1]] += float(_th1o)
                    tau[LEG_CTRL_IDX[b + 2]] += float(_th2o)
                # 贴面：摆腿轮保留小前驱（滚上立面），非自由
                F_w = np.asarray(F_des[leg], dtype=np.float64)
                # v1082: ??/HOVER ? hipx ?? 0?????????????
                # ?? + F_y ????????????v1081 ??????
                # v1092: ?? SWING = ??????????????? PD ??
                # ?? J^T?F_des ????? NMPC ?????????????
                # v1141/v1144: ???????????????v1109 J^T ??
                # ??????????? J^T?F_des ????v1070"?????
                # ???"????????? riser2??
                if leg in (2, 3):
                    _fb2 = R.T @ F_w
                    _fs2 = np.array([float(_fb2[0]), -float(_fb2[2])])
                    _J2 = self.fk.jac(q1, q2)
                    _th1f, _th2f = _J2.T @ _fs2
                    tau[LEG_CTRL_IDX[b + 1]] += float(_th1f)
                    tau[LEG_CTRL_IDX[b + 2]] += float(_th2f)
                tau[LEG_CTRL_IDX[b]] = float(
                    0.30 * F_w[1] + 80.0 * (
                        -0.05 if leg in (0, 1) else 0.05))
                fwd_w = R @ np.array([1.0, 0.0, 0.0])
                fx_fb = float(np.dot(F_w, fwd_w))
                tau[WHEEL_Q_IDX[leg]] = float(np.clip(
                    -self.fk.r * fx_fb, -13.5, 13.5))
            else:
                # 支撑腿：J^T·(R^T·F_des) 力分配——F_des 是作用在 body 的
                # 接触力（向上），与 VMC 约定一致（f_b=R^T·F，f_sag=[fx,-fz]）
                _qp1 = -1.16 if leg in (0, 1) else 1.16
                _qp2 = 2.30 if leg in (0, 1) else -2.30
                F_w = np.asarray(F_des[leg], dtype=np.float64)
                f_b = R.T @ F_w
                fx, fy, fz_up = float(f_b[0]), float(f_b[1]), float(f_b[2])
                f_s = np.array([fx, -fz_up])   # 矢状面 (x, z_down)
                # v1068: 地形阻抗（VMC 同款 kp_h=300）——NMPC F 在过渡态
                # 太小、姿态正则被压过 → 腿泵高发射；阻抗按轮高误差强压
                _pz_d = float(terrain_h[leg]) + r - float(os.environ.get(
                    'S10_NMPC_PRESS', '0.005'))
                _dz_h = _pz_d - float(wheel_xyz[leg, 2])
                _fz_imp = float(os.environ.get(
                    'S10_NMPC_KPH', '300.0')) * _dz_h
                f_s = f_s + np.array([0.0, _fz_imp])
                J = self.fk.jac(q1, q2)
                # v1069: 过伸强位置保持——轮高于地形目标时，J^T 在上折位形
                # 失去权威（轮 1.09/body 0.85 实测），改用关节空间高增益
                # PD 把腿拉回标称弯曲位形（防上折，v1014 思路）
                if float(wheel_xyz[leg, 2]) > _pz_d + 0.02:
                    _kpo = float(os.environ.get('S10_NMPC_KP_OVR', '300.0'))
                    _kdo = float(os.environ.get('S10_NMPC_KD_OVR', '30.0'))
                    tau[LEG_CTRL_IDX[b + 1]] += _kpo * (_qp1 - q1) - _kdo * dq1
                    tau[LEG_CTRL_IDX[b + 2]] += _kpo * (_qp2 - q2) - _kdo * dq2
                th1, th2 = J.T @ f_s
                # v1056: 姿态正则（零空间 PD 拉回蹲姿，VMC 同款）——力控在
                # 近奇异位形失去权威，腿漂到折叠上伸（轮 1.02/body 0.64
                # 卡死实测）；正则把腿拉回标称蹲姿，防奇异漂移
                _kp_pose = float(os.environ.get('S10_NMPC_KP_POSE', '20.0'))
                _kd_pose = float(os.environ.get('S10_NMPC_KD_POSE', '6.0'))
                # v1071: 对侧轴 SWING 时本轴强姿态保持（固定蹲姿）——
                # v1076 IK 姿态版本在真原图更差（东漂+早翻），回退
                # v1100: ???????? SWING ??????????????
                # pz_d???????? body ??????????kp_pose 200 +
                # ???? 300 ??????????real23 ????????
                # 1.1 ???????????? SWING ????????????
                # ?????????v1098 ??????? real29 ?????
                if float(np.max(swing[2:4])) > 0.5 and leg in (0, 1):
                    _kp_pose = float(os.environ.get(
                        'S10_NMPC_KP_POSE_OPP', '200.0'))
                    _kd_pose = float(os.environ.get(
                        'S10_NMPC_KD_POSE_OPP', '20.0'))
                    _rel3 = np.array([wheel_xyz[leg, 0] - hip_w[0],
                                      wheel_xyz[leg, 1] - hip_w[1],
                                      _pz_d - hip_w[2]])
                    _cy3, _sy3 = np.cos(body['yaw']), np.sin(body['yaw'])
                    _relb3 = np.array([_cy3 * _rel3[0] + _sy3 * _rel3[1],
                                       -_sy3 * _rel3[0] + _cy3 * _rel3[1],
                                       _rel3[2]])
                    _relb3[0] = max(float(_relb3[0]), 0.0)
                    _rz3 = float(np.clip(_relb3[2], -0.34, 0.0))
                    _q1p, _q2p = self._ik(
                        float(_relb3[0]), _rz3, q1, q2, leg=leg)
                    _qp1, _qp2 = _q1p, _q2p
                if float(np.max(swing[0:2])) > 0.5 and leg in (2, 3):
                    _kp_pose = float(os.environ.get(
                        'S10_NMPC_KP_POSE_OPP', '200.0'))
                    _kd_pose = float(os.environ.get(
                        'S10_NMPC_KD_POSE_OPP', '20.0'))
                tau[LEG_CTRL_IDX[b + 1]] = float(
                    th1 + _kp_pose * (_qp1 - q1) - _kd_pose * dq1)
                tau[LEG_CTRL_IDX[b + 2]] = float(
                    th2 + _kp_pose * (_qp2 - q2) - _kd_pose * dq2)
                # hipx 侧向力（VMC 同款固定 0.30 杠杆）+ 标称外展
                # v1078 回退：HipX 直接 yaw 误差项过冲（t=9 翻），v1075 基线最优
                tau[LEG_CTRL_IDX[b]] = float(0.30 * fy + 80.0 * (
                    -0.05 if leg in (0, 1) else 0.05))
        # 轮：Pfaffian 驱动——τ 跟随 NMPC 接触力前向分量（受摩擦锥约束），
        # vx PID 只做微调，下限仅防倒转。v1065: 前驱下限-5×4=247N 前向力
        # 作用在轮心（CoM 下方）→ 50Nm 俯仰 → 翘起发射（台架实测）；
        # F_des 前馈的推力受摩擦锥/规划约束，不会发射。
        _dfx = -float(os.environ.get('S10_NMPC_DRIVE_FLOOR', '2.0'))
        _om_cmd = float(cmd.get('omega', 0.0))
        _any_sw = float(np.max(swing)) > 0.5
        fwd_w = R @ np.array([1.0, 0.0, 0.0])
        for leg in range(4):
            wq = float(qvel[WHEEL_QV_IDX[leg]])
            v_wheel = -wq * r
            _sd = -1.0 if leg in (0, 2) else 1.0
            F_w = np.asarray(F_des[leg], dtype=np.float64)
            fx_fb = float(np.dot(F_w, fwd_w))
            _tw = -r * fx_fb
            if swing[leg] > 0.5:
                # 摆腿（贴面滚爬）：保留前向微驱
                _tw += -0.5 * float(os.environ.get(
                    'S10_NMPC_WHEEL_K', '4.0')) * (self._vx_f - v_wheel)
            else:
                _vref = self._vx_f
                if not _any_sw:
                    _vref = self._vx_f + _sd * _om_cmd * self.track_half
                # vx PID 微调（权重 0.3）
                _tw += -0.3 * float(os.environ.get(
                    'S10_NMPC_WHEEL_K', '4.0')) * (_vref - v_wheel)
                # 防倒转下限（小）
                _tw = min(float(_tw), _dfx)
                # yaw 率阻尼 + 航向误差（轮差速）
                _tw += _sd * float(qvel[5]) * float(os.environ.get(
                    'S10_NMPC_YAW_DIFF', '2.0')) * self.track_half
                try:
                    _hdg_e = float(getattr(self.stair, '_hdg_err', 0.0))
                    _ky = float(os.environ.get('S10_NMPC_YAW_ERR_K', '2.0'))
                    _tw -= _sd * _hdg_e * _ky * self.track_half
                except Exception:
                    pass
            tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -48, 48)
        tau[WHEEL_Q_IDX] = np.clip(tau[WHEEL_Q_IDX], -13.5, 13.5)
        return tau

    def _ik(self, xd, zd, q1, q2, leg=None):
        L1, L2 = self.fk.L1, self.fk.L2
        zd_d = -zd
        r2 = min(xd * xd + zd_d * zd_d, (L1 + L2) ** 2 - 1e-6)
        c2 = float(np.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2),
                           -1.0, 1.0))
        q2n = float(np.arccos(c2))
        # v1059: 按腿分镜像分支（后腿 q2 负）——原恒正，后腿 IK 解错
        # 位形、摆腿把轮甩过头（轮 0.99 vs 目标 0.75 实测）
        if leg is not None and leg in (2, 3):
            q2n = -q2n
        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        return q1n, q2n

    # ---------------- 主入口 ----------------
    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):
        self._t += dt
        body = self._body_state(qpos, qvel)
        fwd = np.array([np.cos(body['yaw']), np.sin(body['yaw'])])
        # 速度低通
        _vt = float(os.environ.get('S10_NMPC_VX_TAU', '0.10'))
        self._vx_f += (float(cmd.get('vx', 0.0)) - self._vx_f) * min(1.0, dt / _vt)
        # ModeSequence（布尔抬升轮）
        swing = self._mode_sequence(body['pos'], fwd, wheel_xyz)
        # 轨迹层参考
        ref = self._ref_traj(body, self._vx_f, float(cmd.get('omega', 0.0)),
                             terrain_h)
        # v1055: z_ref 从接管时实际高度 0.5s 斜坡到目标——初始 body 高于
        # 参考时 QP 想下压（轮不能拉）→ 自由落体过冲振荡（台架实测）
        if not hasattr(self, '_z_ref0'):
            self._z_ref0 = float(body['pos'][2])
            self._z_ref_t0 = self._t
        _zr_ramp = float(os.environ.get('S10_NMPC_Z_RAMP', '0.5'))
        _zf = _zr_ramp * 1.0
        _fz = float(np.clip((self._t - self._z_ref_t0) / max(_zf, 1e-3),
                            0.0, 1.0))
        ref['z'] = self._z_ref0 + (ref['z'] - self._z_ref0) * _fz
        # NMPC（20Hz 重解）
        _nmpc_dt = 1.0 / max(self.nmpc_hz, 1e-3)
        if not hasattr(self, '_last_nmpc_t'):
            self._last_nmpc_t = -1e9
        if self._t - self._last_nmpc_t >= _nmpc_dt - 1e-6:
            import time as _tm2
            _t0n = _tm2.perf_counter()
            self._last_nmpc_t = self._t
            self._F_des, self._a_des = self._nmpc(
                body, ref, swing, wheel_xyz, _nmpc_dt)
            self._nmpc_ms = 1e3 * (_tm2.perf_counter() - _t0n)
            self._dbg_fdes = np.asarray(self._F_des).reshape(12)
        else:
            if not hasattr(self, '_F_des'):
                self._F_des = np.array([
                    [0, 0, self.m * self.g / 4.0]] * 4)
                self._a_des = np.zeros(3)
        # WBC（200Hz）
        import time as _tm
        _t0 = _tm.perf_counter()
        tau = self._wbc(qpos, qvel, wheel_xyz, body, self._F_des, swing,
                        cmd, terrain_h, ref, dt)
        if os.environ.get('S10_NMPC_DEBUG', '0') == '1' and \
                int(self._t * 200) % 40 == 0:
            print('[T] nmpc=%.1fms wbc=%.1fms total=%.1fms'
                  % (getattr(self, '_nmpc_ms', 0.0),
                     1e3 * (_tm.perf_counter() - _t0),
                     1e3 * _tm.perf_counter()), flush=True)
        return tau
