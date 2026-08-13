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

# M1leg (2026-08-13): LEG_ATTACH must match the XML joint/body order
# fl, fr, hl, hr (S10.xml: fl_hipx, fr_hipx, hl_hipx, hr_hipx). The old
# table assumed fl, rl, fr, rr -> legs 1 (fr) and 2 (hl) had swapped
# attachment positions (fr placed 0.456m behind at rear-left, hl placed
# 0.456m ahead at front-right). Consequences: front-right swing trigger /
# hip_w 0.45m late -> one-sided front lift -> roll -> launch; rear axle
# targets mirrored -> L/R asymmetry (the whole 155-run wall).
LEG_ATTACH = np.array([
    [0.2277, 0.181191],   # 0 fl: front-left  (+x, +y)
    [0.2277, -0.181191],  # 1 fr: front-right (+x, -y)
    [-0.2277, 0.181191],  # 2 hl: rear-left   (-x, +y)
    [-0.2277, -0.181191], # 3 hr: rear-right  (-x, -y)
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
        self._hover_t = [0.0]*4
        self.swing_to = 1.2
        self.mu = float(os.environ.get('S10_NMPC_MU', '0.8'))
        self.fz_max = float(os.environ.get('S10_NMPC_FZ_MAX', '180.0'))
        self.nmpc_hz = float(os.environ.get('S10_NMPC_HZ', '20.0'))
        self.nmpc_horizon = int(os.environ.get('S10_NMPC_H', '4'))
        # 轨迹层状态
        self._vx_f = 0.0
        self._om_f = 0.0
        self._t = 0.0
        self._sp = [0.0, 0.0, 0.0, 0.0]
        self._sp_top = [0.0, 0.0, 0.0, 0.0]
        self._sw_t0 = [-1e9, -1e9, -1e9, -1e9]
        self._sp_tgt = [-1, -1, -1, -1]
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
        # M1att (2026-08-13): physical pitch/roll via yaw-removed R.
        # Old direct quaternion extraction swaps pitch/roll for a yaw~90deg
        # heading (this map): nose-up reads as roll, side-roll as pitch ->
        # body was held level on stairs + pitch feedforward applied to the
        # lateral axis -> sideways lean (155-run wall root cause).
        _cy, _sy = np.cos(yaw), np.sin(yaw)
        _RzT = np.array([[_cy, _sy, 0.0], [-_sy, _cy, 0.0], [0.0, 0.0, 1.0]])
        _Rp = _RzT @ R
        pitch = float(np.arctan2(-_Rp[2, 0],
                                 np.hypot(_Rp[2, 1], _Rp[2, 2])))
        roll = float(np.arctan2(_Rp[2, 1], _Rp[2, 2]))
        vw = R.T @ np.asarray(qvel[0:3], dtype=np.float64)
        return dict(pos=qpos[0:3], yaw=yaw, roll=roll, pitch=pitch,
                    vx=float(vw[0]), R=R,
                    vel=np.asarray(qvel[0:3], dtype=np.float64),
                    omega=np.asarray(qvel[3:6], dtype=np.float64))

    # ---------------- ModeSequence: lifted wheels (F=0) ----------------
    # M1step (2026-08-13): step-number based swing triggers replace the fixed
    # distance windows. Target riser is explicit: front wheel = next swingable
    # riser (dh>0.085, riser1 pure-roll), rear wheel = current front-axle step.
    # Swing exit uses THIS leg's target riser (wheel past edge + height OK),
    # fixing the old 'nearest-riser switches to next riser after crossing ->
    # geometric exit never fires -> swing drops mid-face -> body collapse' bug.
    # Kept: per-leg independence (diagonal support), G1 body_z gate.
    def _step_of(self, ax_xy):
        """Step number: risers already crossed (tangent d>0.05), incl. riser1."""
        _cur = 0
        for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
            _dd = float(np.dot(
                np.asarray(ax_xy, dtype=np.float64) - _rp[:2], _tng))
            if _dd > 0.05:
                _cur += 1
        return _cur

    def _tgt_riser(self, idx):
        if 0 <= int(idx) < len(self.stair_world):
            return self.stair_world[int(idx)]
        return None

    def _mode_sequence(self, body_pos, fwd, wheel_xyz):
        """Per-leg SWING via step-number trigger.
        Windows: front 0.35 (keeps -0.40 riser1-face boundary clear),
        rear 0.75 (rear hip to front's riser = axle 0.456 + reach 0.15-0.25)."""
        r = self.fk.r
        _perp = np.array([-fwd[1], fwd[0]])
        _steps = [self._step_of(wheel_xyz[leg, :2]) for leg in range(4)]
        _front_step = int(np.min(_steps[0:2]))   # trailing side of front axle
        # M1ax (2026-08-13): front-axle SYMMETRIC trigger — both front hips'
        # d to their target riser, triggered together. Per-leg triggers
        # fired 0.1-0.2s apart (left/right leg reach asymmetry) -> one front
        # wheel climbed alone (L0 0.61 / L1 0.77) -> roll -0.87 -> L3 throw.
        # The body hold (M1reach) provides the support the old axis-sync
        # lacked; exits stay per-leg (each wheel crosses the edge itself).
        _front_d = [1e9, 1e9]
        for _fl in (0, 1):
            _fh = (body_pos[:2] + fwd * LEG_ATTACH[_fl][0]
                   + _perp * LEG_ATTACH[_fl][1])
            _fi = _steps[_fl]
            while (_fi < len(self.stair_world)
                   and float(self.stair_world[_fi][3]) <= 0.085):
                _fi += 1
            _ft = self._tgt_riser(_fi)
            if _ft is not None and float(_ft[3]) > 0.085:
                _front_d[_fl] = float(np.dot(_fh - _ft[0], _ft[1]))
        for leg in range(4):
            # trigger distance from HIP (same convention as M1eee4/M1mmm3:
            # wheel-based distances fire 0.15-0.25 m early -> front F=0 too
            # long -> NMPC primal infeasible -> garbage F -> leg throw)
            _hip_xy = (body_pos[:2] + fwd * LEG_ATTACH[leg][0]
                       + _perp * LEG_ATTACH[leg][1])
            _wxy = wheel_xyz[leg, :2]
            _cur = _steps[leg]
            if leg in (2, 3):
                # M1ph2 (2026-08-13): rear arms only after the front-axle
                # swing EXITS. Arming at front_step>=2 while the front legs
                # are still swinging gave sp=[1,1] -> zero support -> body
                # sags 0.73->0.68 -> front legs flip -> pos_lift throw.
                if float(np.max(self._sp[0:2])) <= 0.5:
                    _tgt_idx = _front_step - 1   # rear target = front axle step
                else:
                    _tgt_idx = -9                # wait for the front axle
            else:
                _tgt_idx = _cur                  # front target = next swingable
                while (_tgt_idx < len(self.stair_world)
                       and float(self.stair_world[_tgt_idx][3]) <= 0.085):
                    _tgt_idx += 1
            _tgt = self._tgt_riser(_tgt_idx)
            _d, _top = 1e9, 0.0
            if _tgt is not None and float(_tgt[3]) > 0.085:
                _d = float(np.dot(_hip_xy - _tgt[0], _tgt[1]))
                _top = float(_tgt[4])
            _wz = float(wheel_xyz[leg, 2])
            # G1 gate -0.10 (riser3 needs body 0.822; 0.81 fails -> no swing)
            _bz_ok = float(body_pos[2]) > float(_top) + r - 0.10
            if self._sp[leg] <= 0.0:
                _below_top = _wz < float(_top) + r - 0.02
                _sw_win = 0.75 if leg in (2, 3) else 0.50
                _d_use = _d
                _tripod_ok = float(np.sum(self._sp)) <= 0.5
                if (_tripod_ok and -_sw_win < _d_use < 0.05
                        and _bz_ok and _below_top):
                    self._sp[leg] = 1.0
                    self._sp_top[leg] = float(_top)
                    self._sp_tgt[leg] = int(_tgt_idx)
                    self._sw_t0[leg] = self._t
            else:
                # exit: this leg's target riser crossed (wheel past edge)
                # and wheel height reached; use wheel xy (physical contact).
                _tt = self._tgt_riser(self._sp_tgt[leg])
                _wd = 1e9
                if _tt is not None:
                    _wd = float(np.dot(_wxy - _tt[0], _tt[1]))
                if _wd > 0.05 and _wz >= self._sp_top[leg] + r - 0.01:
                    self._sp[leg] = 0.0
                elif self._t - self._sw_t0[leg] > self.swing_to:
                    self._sp[leg] = 0.0
        swing = np.array(self._sp, dtype=np.float64)
        self._dbg_phases = (float(np.max(swing[0:2])), float(np.max(swing[2:4])))
        return swing

    def _ref_traj(self, body, vx_cmd, om_cmd, terrain_h):
        """台阶推进参考：vx 插值、body z 跟轮下地形均值+偏移、pitch 随坡度、
        heading 锁楼梯切线（v1034 思路）。"""
        r = self.fk.r
        _z_geo = []
        _sw_d = max(self.swing_d, 0.15)
        for leg in range(4):
            gt = float(terrain_h[leg])
            # M1ooo2: 每轴（前/后轴）用轴位置定斜坡相位
            # （每腿髋位在 yaw 偏差下左右相位不同
            # →左右目标不对称→单侧泵高实测；
            # 轴位置左右对称、前后正确）
            _ax_off = 0.2277 if leg in (0, 1) else -0.2277
            _ax_xy = body['pos'][:2] + (
                body['R'] @ np.array([_ax_off, 0.0, 0.0]))[:2]
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                if _dhv <= 0.085:
                    continue
                _dd = float(np.dot(
                    _ax_xy - _rp[:2], _tng))
                # M1sss: z 参考沿摆动窗 smoothstep 爬升（原 0.4m
                # 提前跳到台面顶→body z 过早抬 0.125m→
                # 摆腿期泵高/发射实测）
                if _dd > 0.05:
                    gt = max(gt, float(_top))
                elif _dd > -_sw_d:
                    _zbt = float(_top - _dhv)
                    _t = float(np.clip((_dd + _sw_d) / max(_sw_d, 1e-3), 0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    gt = max(gt, _zbt + _dhv * _ss)
            _z_geo.append(gt + r)
        # #4(???): body z ????????lerp(??????)+?????
        # ?? 0.25??? 0.22 ?????body ? 0.66??? drop<0.1 ????
        # 0.25 ?? body 0.80+?drop>=0.15??Z_OFF ???????????
        z_ref = float(np.mean(_z_geo)) + float(
            os.environ.get('S10_NMPC_Z_OFF', '0.25'))
        # #5l(????????): ????? riser 2m~0.4m ? z_ref ? 0.05??
        # ????????? 0.86??? riser1 ??? drop ?? >=0.15 ?
        # ???????? riser1 ? 0.14??????????????
        if self.stair_world:
            _f0 = self.stair_world[0][0]
            _t0 = self.stair_world[0][1]
            _d_first = float(np.dot(
                np.asarray(body['pos'][:2], dtype=np.float64)
                + np.array([np.cos(body['yaw']), np.sin(body['yaw'])]) * 0.228
                - _f0, _t0))
            if -2.0 < _d_first < -0.4:
                z_ref += 0.05
        _pitch_ref = float(np.arctan2(
            float(np.mean(_z_geo[0:2])) - float(np.mean(_z_geo[2:4])),
            0.456))
        # 楼梯切线 heading
        hdg = float(np.arctan2(self.stair_world[0][1][1],
                               self.stair_world[0][1][0])) \
            if self.stair_world else float(body['yaw'])
        return dict(vx=vx_cmd, z=z_ref, pitch=_pitch_ref, hdg=hdg,
                    wz_geo=np.array(_z_geo, dtype=np.float64))

    # ---------------- NMPC：SRBD 接触力优化（20Hz）----------------
    def _nmpc(self, body, ref, swing, wheel_xyz, dt_nmpc, terrain_h=None):
        """SRBD 轨迹 QP（M1 多 knot）：每 knot [F(12),a(3),p(3),v(3)]，
        状态传播硬等式 + 常数 a_des 跟踪（m23 语义）+ 力平滑。
        v2026 M1g 修复 SRBD 三轴等式（原把 x/y/z 全塞一行→动力学无约束）。"""
        m, g = self.m, self.g
        R = body['R']
        K = int(os.environ.get('S10_NMPC_HORIZON', '8'))
        dt = max(dt_nmpc, 1e-3)
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
        w_fr = float(os.environ.get('S10_NMPC_WFR', '0.02'))
        w_s = float(os.environ.get('S10_NMPC_WS', '0.02'))
        w_p = float(os.environ.get('S10_NMPC_WP', '0.05'))
        w_v = float(os.environ.get('S10_NMPC_WV', '0.10'))
        F_ref = np.zeros(12)
        _n_cont = max(float(np.sum(swing <= 0.5)), 1.0)
        for i in range(4):
            F_ref[3*i+2] = m * g / _n_cont * (
                1.0 if swing[i] <= 0.5 else 0.2)
        fwd_w = R @ np.array([1.0, 0.0, 0.0])
        al_des = np.zeros(3)
        al_des[1] = kp_p * (ref['pitch'] - body['pitch']) \
            - kd_p * float(np.dot(R[:, 1], body['omega']))
        _ff_sw = 1.5
        if _ff_sw > 0.0:
            if float(np.max(swing[2:4])) > 0.5:
                al_des[1] -= _ff_sw
            elif float(np.max(swing[0:2])) > 0.5:
                al_des[1] += _ff_sw
        al_des[2] = kp_y * (ref['hdg'] - body['yaw']) \
            - kd_y * float(np.dot(R[:, 2], body['omega']))
        # M1rl2 (2026-08-13): during any swing, scale down the ROLL
        # correction (the QP realizes it with the rear legs' LATERAL forces,
        # eating the friction budget -> forward F_x starves -> stall at the
        # riser face -> tip). The roll drifts a little; the advance breaks
        # the stall. On flat (no swing) the full gain holds.
        al_des[0] = -18.0 * body['roll']
        al_des[0] -= 12.0 * float(np.dot(R[:, 0], body['omega']))
        if float(np.max(swing)) > 0.5:
            _al_lim = 30.0
            al_des[1] = float(np.clip(al_des[1], -_al_lim, _al_lim))
            if float(body['pitch']) < -0.55:
                al_des[1] += 20.0 * (-0.55 - float(body['pitch']))
        _w_b = R.T @ np.asarray(body['omega'], dtype=np.float64)
        M_des = R @ (self.I_body @ al_des
                     + np.cross(_w_b, self.I_body @ _w_b))
        # M1ggg: 角状态——线性化 ωxIω 于当前 ω
        _w0 = np.asarray(body['omega'], dtype=np.float64)
        _Ix, _Iy, _Iz = np.diag(self.I_body)
        _Jw = np.zeros((3, 3))
        _Jw[0, 1] = (_Iz - _Iy) * _w0[2]
        _Jw[0, 2] = (_Iz - _Iy) * _w0[1]
        _Jw[1, 0] = (_Ix - _Iz) * _w0[2]
        _Jw[1, 2] = (_Ix - _Iz) * _w0[0]
        _Jw[2, 0] = (_Iy - _Ix) * _w0[1]
        _Jw[2, 1] = (_Iy - _Ix) * _w0[0]
        _cw = np.cross(_w0, self.I_body @ _w0) - _Jw @ _w0
        r_w = wheel_xyz - body['pos']
        A_m = np.zeros((3, 12))
        for i in range(4):
            rc = r_w[i]
            A_m[0, 3*i+1] = -rc[2]
            A_m[0, 3*i+2] = rc[1]
            A_m[1, 3*i+0] = rc[2]
            A_m[1, 3*i+2] = -rc[0]
            A_m[2, 3*i+0] = -rc[1]
            A_m[2, 3*i+1] = rc[0]
        for i in range(4):
            if swing[i] > 0.5 and i in (0, 1):
                A_m[:, 3*i:3*i+3] = 0.0
        nk = 38  # M1ggg+ooo+zzz: [...,wv(4),wz_sw(4)]
        n = nk * K
        P = np.zeros((n, n))
        q = np.zeros(n)
        pos0 = np.asarray(body['pos'], dtype=np.float64)
        vel0 = np.asarray(body['vel'], dtype=np.float64)
        # M1www: 抗发射用线性垂直速度（原用角速度
        # =yaw 率，z 阻尼和抗发射全部错位）
        _vz_b = float(np.dot(R[:, 2], body['vel']))
        _zlo = -10.0 if _vz_b > 0.5 else -4.0
        _any_sw = float(np.max(swing)) > 0.5
        a_des = np.zeros(3)
        a_des[2] = float(np.clip(kp_z * (ref['z'] - pos0[2])
                                - kd_z * _vz_b,
                                _zlo, 8.0))
        v_w0 = fwd_w * body['vx']
        a_des[0:2] = kp_vx * (ref['vx'] * fwd_w[0:2] - v_w0[0:2])
        if _any_sw:
            _n2 = float(np.dot(fwd_w[0:2], fwd_w[0:2]))
            if _n2 > 1e-6:
                a_des[0:2] = (float(np.dot(a_des[0:2], fwd_w[0:2])) / _n2) * fwd_w[0:2]
        # M1xxx3: 摆动轮几何目标（与 WBC swing_tgt_z 同款 smoothstep）
        _wz_sw_tgt = np.zeros(4)
        for leg in range(4):
            _ax_idx = (0, 1) if leg in (0, 1) else (2, 3)
            _ax_xy = np.mean([wheel_xyz[_i, :2] for _i in _ax_idx], axis=0)
            _best_d, _best = 1e9, None
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                if _dhv <= 0.085:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))
                _sw_win_t = 0.75 if leg in (2, 3) else 0.45
                if -_sw_win_t < _dd < 0.05 and abs(_dd) < abs(_best_d):
                    _best_d, _best = _dd, (_rp, _tng, _dhv, _top)
            if _best is not None:
                (_rp, _tng, _dhv, _top) = _best
                _z_bot = float(_top - _dhv)
                _d_w = _best_d
                _rr = self.fk.r
                # M1arc2 (2026-08-13): FRONT swing-wheel target on the face
                # arc (last R) like the rear. The window smoothstep lifted
                # the NMPC target from -0.35 -> the WBC wheel unloaded early
                # -> drive lost -> stall at the face (front held 0.61 for
                # 5s). The wheel must drive until d > -R, then lift.
                _t = float(np.clip((_d_w + _rr) / max(_rr, 1e-3), 0.0, 1.0))
                _ss = _t * _t * (3.0 - 2.0 * _t)
                _zc = _z_bot + _rr + _dhv * _ss
                _wz_sw_tgt[leg] = min(_zc, _top + _rr + 0.005)
            else:
                _wz_sw_tgt[leg] = float(wheel_xyz[leg, 2])
        for k in range(K):
            f0 = nk * k
            P[f0+12:f0+15, f0+12:f0+15] += 2.0 * w_a * np.eye(3)
            q[f0+12:f0+15] += -2.0 * w_a * a_des
            Fs = np.arange(f0, f0+12)
            P[np.ix_(Fs, Fs)] += 2.0 * (
                w_f * np.eye(12) + w_m * (A_m.T @ A_m) + w_fr * np.eye(12))
            q[Fs] += -2.0 * (w_m * (A_m.T @ M_des) + w_fr * F_ref)
            # M1ggg: 角跟踪代价 roll->0, pitch->ref, yaw->hdg
            _th_ref = np.array([0.0, ref['pitch'], ref['hdg']])
            for _j in range(3):
                P[f0+27+_j, f0+27+_j] += 2.0 * w_m * 200.0
                q[f0+27+_j] += -2.0 * w_m * 200.0 * _th_ref[_j]
            p_free = pos0 + vel0 * (k * dt)
            p_idx = [f0+15, f0+16, f0+17]
            v_idx = [f0+18, f0+19, f0+20]
            P[np.ix_(p_idx, p_idx)] += 2.0 * w_p * np.eye(3)
            q[p_idx] += -2.0 * w_p * p_free
            P[np.ix_(v_idx, v_idx)] += 2.0 * w_v * np.eye(3)
            q[v_idx] += -2.0 * w_v * vel0
            # M1ooo: wheel Pfaffian - wv(4) linear wheel speed per knot
            # stance: (wv_i - fwd_w.v)^2 rolling consistency; swing: light
            # tracking of commanded vx; result feeds WBC wheel PID ref
            w_wheel = float(os.environ.get('S10_NMPC_WWHEEL', '30.0'))
            _wv_idx = [f0+30, f0+31, f0+32, f0+33]
            for _wi in range(4):
                if swing[_wi] > 0.5:
                    P[_wv_idx[_wi], _wv_idx[_wi]] += 2.0 * 0.1 * w_wheel
                    q[_wv_idx[_wi]] += -2.0 * 0.1 * w_wheel * ref['vx']
                else:
                    P[_wv_idx[_wi], _wv_idx[_wi]] += 2.0 * w_wheel * 1.02
                    P[np.ix_(v_idx, v_idx)] += 2.0 * w_wheel * (np.outer(fwd_w, fwd_w) + 0.02 * np.eye(3))
                    for _j in range(3):
                        P[_wv_idx[_wi], f0+18+_j] -= 2.0 * w_wheel * fwd_w[_j]
                        P[f0+18+_j, _wv_idx[_wi]] -= 2.0 * w_wheel * fwd_w[_j]
            if k >= 1:
                f1w = nk * (k-1)
                for _wi in range(4):
                    P[_wv_idx[_wi], _wv_idx[_wi]] += 2.0 * 0.2 * w_wheel
                    P[f1w+30+_wi, f1w+30+_wi] += 2.0 * 0.2 * w_wheel
                    P[f1w+30+_wi, _wv_idx[_wi]] -= 2.0 * 0.2 * w_wheel
                    P[_wv_idx[_wi], f1w+30+_wi] -= 2.0 * 0.2 * w_wheel
            # M1xxx3: 摆动轮 z 状态 wz_sw(4)，与 body z 耦合
            _wz_idx = [f0+34, f0+35, f0+36, f0+37]
            for _wi in range(4):
                P[_wz_idx[_wi], _wz_idx[_wi]] += 2.0 * 1.0
                q[_wz_idx[_wi]] += -2.0 * 1.0 * _wz_sw_tgt[_wi]
            if k >= 1:
                f1 = nk * (k-1)
                P[np.ix_(Fs, Fs)] += 2.0 * w_s * np.eye(12)
                P[np.ix_(np.arange(f1, f1+12), np.arange(f0, f0+12))] -= 2.0 * w_s * np.eye(12)
                P[np.ix_(np.arange(f0, f0+12), np.arange(f1, f1+12))] -= 2.0 * w_s * np.eye(12)
        neq = 3*K + 3*K + 6 + 6*(K-1) + 6 + 6*(K-1)
        Ae = np.zeros((neq, n))
        be = np.zeros(neq)
        r_mom = 3*K
        r_lin_init = 6*K
        r_lin_prop = 6*K + 6
        r_ang_init = 6*K + 6 + 6*(K-1)
        r_ang_prop = r_ang_init + 6
        for k in range(K):
            f0 = nk * k
            Ae[3*k+0, f0+0:f0+12:3] = -1.0
            Ae[3*k+1, f0+1:f0+12:3] = -1.0
            Ae[3*k+2, f0+2:f0+12:3] = -1.0
            Ae[3*k+0, f0+12] = m
            Ae[3*k+1, f0+13] = m
            Ae[3*k+2, f0+14] = m
            be[3*k+2] = -m * g
            for _j in range(3):
                Ae[r_mom+3*k+_j, f0+21+_j] = self.I_body[_j, _j]
                Ae[r_mom+3*k+_j, f0+0:f0+12] += -A_m[_j, :]
                Ae[r_mom+3*k+_j, f0+24:f0+27] += -_Jw[_j, :]
            be[r_mom+3*k:r_mom+3*k+3] = _cw
            if k == 0:
                for j in range(3):
                    Ae[r_lin_init+j, f0+15+j] = 1.0
                    be[r_lin_init+j] = pos0[j]
                    Ae[r_lin_init+3+j, f0+18+j] = 1.0
                    be[r_lin_init+3+j] = vel0[j]
                    Ae[r_ang_init+j, f0+24+j] = 1.0
                    be[r_ang_init+j] = _w0[j]
                    Ae[r_ang_init+3+j, f0+27+j] = 1.0
                    be[r_ang_init+3+j] = body['roll'] if j == 0 else (
                        body['pitch'] if j == 1 else body['yaw'])
            else:
                rp = r_lin_prop + 6*(k-1)
                ra = r_ang_prop + 6*(k-1)
                f1 = nk * (k-1)
                for j in range(3):
                    Ae[rp+j, f0+18+j] = 1.0
                    Ae[rp+j, f1+18+j] = -1.0
                    Ae[rp+j, f1+12+j] = -dt
                    Ae[rp+3+j, f0+15+j] = 1.0
                    Ae[rp+3+j, f1+15+j] = -1.0
                    Ae[rp+3+j, f1+18+j] = -dt
                    Ae[ra+j, f0+24+j] = 1.0
                    Ae[ra+j, f1+24+j] = -1.0
                    Ae[ra+j, f1+21+j] = -dt
                    Ae[ra+3+j, f0+27+j] = 1.0
                    Ae[ra+3+j, f1+27+j] = -1.0
                    Ae[ra+3+j, f1+24+j] = -dt

        rows = []
        ub = []
        # M1yyy2: 左右 F_z 差异绑定 ±30N（原 QP 极端分配
        # 一侧 170-180N/另侧 12-28N→低支撑侧轮抬起→
        # 单侧泵高实测；±30N 允许正常 roll 修正）
        # M1mmm3: rows/knot 随同态对数变化（F_z 对称约束条件化）
        _same_pairs = sum(1 for _pr in ((0, 1), (2, 3))
                         if (swing[_pr[0]] > 0.5) == (swing[_pr[1]] > 0.5))
        _rpk = 46 + 4 * _same_pairs
        for k in range(K):
            f0 = nk * k
            for i in range(4):
                e = np.zeros(n)
                e[f0+3*i+2] = -1.0
                rows.append(e); ub.append(0.0)
                if swing[i] <= 0.5:
                    # M1fwd2 (2026-08-13): stance legs plan NO backward
                    # F_x (>=0). The QP realized the pitch moment with the
                    # rear backward F_x, the WBC drops it (max(fx,0)) ->
                    # plan/execution mismatch -> roll/yaw uncontrolled ->
                    # right wheels thrown. Planning F_x>=0 keeps the plan
                    # consistent; the pitch moment comes from the F_z.
                    e = np.zeros(n)
                    e[f0+3*i+0] = -1.0
                    rows.append(e); ub.append(0.0)
                for j in range(2):
                    e = np.zeros(n)
                    e[f0+3*i+j] = 1.0; e[f0+3*i+2] = -self.mu
                    rows.append(e); ub.append(0.0)
                    e = np.zeros(n)
                    e[f0+3*i+j] = -1.0; e[f0+3*i+2] = -self.mu
                    rows.append(e); ub.append(0.0)
                e = np.zeros(n)
                e[f0+3*i+2] = 1.0
                rows.append(e); ub.append(self.fz_max)
            for i in range(12):
                e = np.zeros(n)
                e[f0+i] = 1.0
                rows.append(e); ub.append(1e9)
            e = np.zeros(n)
            e[f0+14] = 1.0
            # M1www: 快速上升时上界设 0（原恒 8，
            # QP 满足上推力→腿泵高实测）
            # M1soft: no body-z slam while wheels are swinging (was the
            # launch driver: 8 m/s2 up + swing lift + pos_lift -> airborne)
            _az_max = 0.0 if _vz_b > 0.5 else (3.0 if _any_sw else 8.0)
            rows.append(e); ub.append(_az_max)
            e = np.zeros(n)
            e[f0+14] = -1.0
            rows.append(e); ub.append(-_zlo)
            for _wi in range(4):
                e = np.zeros(n)
                e[f0+30+_wi] = 1.0
                rows.append(e); ub.append(12.0)
                e = np.zeros(n)
                e[f0+30+_wi] = -1.0
                rows.append(e); ub.append(12.0)
            # M1xxx3: body z ≥ 摆动轮 z + 0.05（计划层禁折叠几何）
            for _wi in range(4):
                if swing[_wi] > 0.5:
                    e = np.zeros(n)
                    e[f0+17] = -1.0
                    e[f0+34+_wi] = 1.0
                    rows.append(e); ub.append(-0.05)
            # M1mmm3: F_z 对称只在左右同态时生效（轮级
            # swing 后单轮摆动时不能强迫支撑轮 F_z=0）
            for _pr in ((0, 1), (2, 3)):
                if (swing[_pr[0]] > 0.5) == (swing[_pr[1]] > 0.5):
                    for _sgn in (1.0, -1.0):
                        e = np.zeros(n)
                        e[f0+3*_pr[0]+2] = _sgn
                        e[f0+3*_pr[1]+2] = -_sgn
                        # M1mmm4: F_z 对称松宽到 ±80（原 ±30 限制
                        # roll 修正→roll 增长→后轮折叠实测）
                        rows.append(e); ub.append(80.0)
        A = np.vstack([Ae] + rows)
        l = np.hstack([be] + [-1e9] * len(rows))
        u = np.hstack([be] + ub)
        for k in range(K):
            f0 = nk * k
            for i in range(4):
                base = neq + (len(rows) // K) * k + 24 + 3 * i
                if swing[i] > 0.5:
                    if i in (0, 1):
                        l[base+0] = 0.0; u[base+0] = 0.0
                        l[base+1] = 0.0; u[base+1] = 0.0
                        l[base+2] = 0.0; u[base+2] = 0.0
                    else:
                        l[base+2] = float(os.environ.get(
                            'S10_NMPC_REAR_SWING_FZ_MIN', '46.0'))
                        u[base+2] = self.fz_max
                elif float(np.max(swing[0:2])) > 0.5 and i in (2, 3):
                    l[base+2] = float(os.environ.get(
                        'S10_NMPC_STANCE_FZ_MIN', '95.0'))
                    u[base+2] = self.fz_max
                elif float(np.max(swing[2:4])) > 0.5 and i in (0, 1):
                    l[base+2] = float(os.environ.get(
                        'S10_NMPC_STANCE_FZ_MIN', '95.0'))
                    u[base+2] = self.fz_max
        import osqp
        from scipy import sparse
        # M1step2: rows/knot varies with swing pattern (_same_pairs) ->
        # osqp.update() rejects row-count changes and leaves stale mixed
        # data -> primal infeasible -> garbage F -> leg throw. Rebuild the
        # problem whenever n or the constraint-row count changes.
        _nrows = len(rows)
        if (self._nmpc_prob is None
                or getattr(self, '_nmpc_nrows', -1) != _nrows
                or getattr(self, '_nmpc_nvars', -1) != n):
            prob = osqp.OSQP()
            prob.setup(P=sparse.csc_matrix(P), q=q,
                       A=sparse.csc_matrix(A), l=l, u=u,
                       verbose=False, eps_abs=1e-3, eps_rel=1e-3,
                       max_iter=800, polish=False)
            self._nmpc_prob = prob
            self._nmpc_nrows = _nrows
            self._nmpc_nvars = n
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
            self._wv_des = np.clip(x[30:34], -12.0, 12.0)
            self._wz_sw_des = np.clip(x[34:38], 0.0, 2.5)
            if os.environ.get('S10_NMPC_DEBUG', '0') == '1':
                print('[NMPC] t=%.2f F=%s a=%s ades=%s Mdes=%s wv=%s st=%s dt=%.1fms K=%d'
                      % (self._t, np.round(F, 1).tolist(),
                         np.round(a, 2).tolist(),
                         np.round(a_des, 2).tolist(),
                         np.round(M_des, 1).tolist(),
                         np.round(self._wv_des, 2).tolist(),
                         res.info.status,
                         1e3 * (_t.perf_counter() - _t0), K), flush=True)
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
                # M1vvv3: 触发用单轮（对角支撑），目标用轴均值
                # （左右同目标防 roll；单轮目标在左右独立
                # swing 时目标不同→侧翻实测）
                _ax_xy = wheel_xyz[leg, :2]
                _best_d, _best = 1e9, None
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    if _dhv <= 0.085:
                        continue
                    _dd = float(np.dot(_ax_xy - _rp, _tng))
                    # v2026(M1u): 目标窗负侧 -0.35->-0.45——触发用前导轮
                    # (±0.181 偏移)，目标用轴均值，均值常比前导轮远 0.1m；
                    # 窗不匹配→swing 有标志无目标→腿悬空无控制→登顶发射
                    _sw_win_t = 0.75 if leg in (2, 3) else 0.45
                    if -_sw_win_t < _dd < 0.05 and abs(_dd) < abs(_best_d):
                        _best_d, _best = _dd, (_rp, _tng, _dhv, _top)
                if _best is not None:
                    (_rp, _tng, _dhv, _top) = _best
                    _z_bot = float(_top - _dhv)
                    _d_w = _best_d
                    # v1058: 爬升斜坡（原 sqrt(R²-d²) 是下降滚动弧，目标低于
                    # 地面 0.55 → 轮目标乱摆发射）；正确：地面+r → 台面顶+r
                    # 随窗 smoothstep（位置基 _face_place_z 同款，已验证）
                    # v1165: ???? 0.3??? 10cm ?????0.15 ?????
                    # ?????0.7 ???body ? 0.75??0.3 ?? v1159 ?
                    # body 0.86 ???????
                    # v2026(M1t): 后轴贴面弧窗 d∈[-R,0]（原 0.3 斜坡在棱前
                    # 0.245m 完成全抬→后轮提前悬空→登顶发射；单点耦合失败，
                    # M1s 前轮已稳定，后轴独立修复）
                    # M1arc2 (2026-08-13): FRONT swing target also on the
                    # face arc (last R). The window smoothstep (from -0.35)
                    # lifted the target while the wheel was still rolling the
                    # riser1 top -> wheel unloaded early -> drive lost ->
                    # stall at the face (front held 0.61 for 5s). The rear
                    # already uses the face arc; the front must too: wheels
                    # drive until d > -R, then lift over the face.
                    _t = float(np.clip(
                        (_d_w + self.fk.r) / max(self.fk.r, 1e-3),
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
            # M1hold2: swing control also engages when the wheel is
            # AIRBORNE above its terrain (overshoot case). The old
            # target>wheel clause skipped it -> an overshooting wheel fell
            # to the stance branch, the force-based hold never pulled it
            # back to the target -> front leg flipped up (L0 0.90/pz -0.28)
            # and the front swing never exited -> both axles swung.
            # M1axl (2026-08-13): the FRONT swing control engages BOTH front
            # legs together (axle-mean wheel z). Per-leg engagement lifted
            # one wheel first (L0 0.73 / L1 0.62) -> roll -1.48. The rear
            # stays per-leg (its wheels are farther apart in phase).
            _wz_ctl = float(wheel_xyz[leg, 2])
            if sl > 0.5 and leg in swing_tgt_z and (
                    float(swing_tgt_z[leg]) > _wz_ctl + 0.005
                    or _wz_ctl
                    > float(terrain_h[leg]) + r + 0.005):
                # 摆腿（贴面爬升）：位置 PD + 小支撑力（F_des 的 F_z 部分）
                _wzs = getattr(self, '_wz_sw_des', None)
                wz_t = float(_wzs[leg]) if (_wzs is not None) else swing_tgt_z[leg]
                # M1cat (2026-08-13): front-axle catch-up. The axle-mean
                # engagement kept both wheels in swing but their heights
                # diverged (L0 0.77 / L1 0.65 at t=7) -> roll -3.13 tip.
                # The lagging front wheel's target is raised toward the
                # leader; the leader stays on the face arc (the overshoot
                # damping returns it). Rear stays per-leg.
                # v1079(方向1): HOVER 期前轮目标 = body 相对 drop（正 drop
                # 恒定，轮随 body 平移），钳在台面顶+半径以上不插台面
                # v1089: ???????????????????-2cm?????
                # ???+R????????????????????????
                # J^T ??????????? body ???real19 ?? body 0.63?
                # ? 0.75 ????????????? z ???????????
                # v1146: ??????v1085b ?????????????-2cm?
                # ?? SWING ? body ??????? 0.64??? 0.747 ???
                # ? ????????real78 ?? 0.91 ???????????
                # ??????? J^T ???????
                # M1nnn3: 摆腿目标距髋 0.12m（原 0.02→轮目标
                # 贴髋时 IK 必须折叠；0.12 保持腿伸展，
                # body 由支撑轮推升，摆腿轮随之升高）
                wz_t = min(wz_t, float(hip_w[2]) - 0.12)

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
                # M1rch (2026-08-13): clamp the swing drop to the reach at
                # the current forward reach. (px 0.3, drop 0.34) is
                # unreachable (0.45 > leg 0.36) -> saturated PD -> rear
                # throw (L2 1.26). The wheel must come back under the hip as
                # the target rises.
                _Lmx = self.fk.L1 + self.fk.L2
                _pz_mx = float(np.sqrt(max(
                    _Lmx ** 2 - float(relb[0]) ** 2 - 1e-3, 0.02)))
                _rz = float(max(_rz, -(_pz_mx - 0.01)))
                q1t, q2t = self._ik(float(relb[0]), _rz, q1, q2, leg=leg)
                # v1075: 后轴 SWING 低增益（少抬多滚）——前轮已证明贴面滚爬
                # 能越阶；后轴强位置引导会把 body 顶起俯仰 → 前腿上折。
                # 后轴 KP 40 让轮主要靠 F_z≥46 滚上立面，俯仰小
                # M1fff4: 力基摆动保持——轮在计划高度由 J^T 力钉住
                # （位置 PD 欠阻尼过冲；力控只在轮高偏离计划时作用）
                # M1jjj4: 力基摆动 kf=300 + 轮垂直速度阻尼
                # （kf 高升力快，速度阻尼衰减过冲）
                _wz_e = float(wheel_xyz[leg, 2]) - wz_t
                _wz_dot = float(_Jf[1, 0] * dq1 + _Jf[1, 1] * dq2) if hasattr(self, '_Jf') else 0.0
                _wz_dot = float(self.fk.jac(q1, q2)[1, 0] * dq1 + self.fk.jac(q1, q2)[1, 1] * dq2)
                # M1soft (2026-08-13): lift force 300->100. The saturated
                # body-z push + swing lift launches the whole robot airborne
                # (fn=[0,0,0,0], wheels 0.80-0.87). Gentler lift avoids the
                # impulse; speed damping retained.
                # M1rlift (2026-08-13): rear swing lift 100->50. The rear
                # legs sit at px~0.3 (horizontal) -> J^T maps the vertical
                # lift into a whip (L3 0.87 overshoot, side lean -0.6).
                # Gentler rear lift; the wheels climb by rolling + assist.
                # M1psw (2026-08-13): swing execution via POSITION PD on the
                # continuous IK target (the literature ETH/IIT approach).
                # The force-based hold (J^T of the height error) whips the
                # extended leg (L3 -> 1.0) because the force maps through the
                # near-horizontal Jacobian. q1t/q2t preserve the forward
                # reach and M1ik keeps the branch continuous; the PD tracks
                # the QP's face-arc z target without the J^T amplification.
                _kd_sw = 15.0 + 80.0 * float(_wz_e > 0.005)
                tau[LEG_CTRL_IDX[b + 1]] = 80.0 * (q1t - q1) - _kd_sw * dq1
                tau[LEG_CTRL_IDX[b + 2]] = 80.0 * (q2t - q2) - _kd_sw * dq2
                # 贴面：摆腿轮保留小前驱（滚上立面），非自由
                F_w = np.asarray(F_des[leg], dtype=np.float64)
                tau[LEG_CTRL_IDX[b]] = float(
                    0.30 * F_w[1] + 80.0 * (
                        -0.05 if leg in (0, 1) else 0.05))
                fwd_w = R @ np.array([1.0, 0.0, 0.0])
                fx_fb = float(np.dot(F_w, fwd_w))
                tau[WHEEL_Q_IDX[leg]] = float(np.clip(
                    -self.fk.r * fx_fb, -13.5, 13.5))
            else:
                # M1ooo3: 前轮摆动期后轴支撑轮纯位置控制抬 body
                # （力控在短腿位形推不动 body；IK 拉伸后腿直接举 body）
                # M1ppp3: 前轮登顶（wz>0.75）时后轴位置控制抬 body
                # （原依赖 swing 状态但登顶期 G1 门控不满
                # 足→swing 惰性→未触发；直接按前轮高度）
                _fr_sw = float(np.max(swing[0:2])) > 0.5
                _rr_sw = float(np.max(swing[2:4])) > 0.5
                _wz_f = float(np.mean(wheel_xyz[0:2, 2]))
                # M1qqq3: 逐腿判断（该后腿不摆动且前轮登顶）
                # M1uuu3: 所有 stance 腿参与抬 body（前腿在 riser2 顶
                # drop 0.19 可达，能撑 body 到 0.94+；原只后腿抬→
                # 后腿 drop 0.28+ 时 J^T 变弱推不动停滞实测）
                # M1hold (2026-08-13): hold body height on the approach too.
                # Legs fold to pz~0.075 (J vertical authority ~0.03) during the
                # stair braking -> body collapses 0.78->0.64 -> G1 gate marginal
                # -> single-leg swing -> saturated a_z=8 -> launch. Fire the
                # position-lift branch whenever the body is below its ref.
                # M1gnd (2026-08-13): pos_lift is a pure IK push that bypasses
                # the _hover unload -> airborne wheels get whipped upward ->
                # whole robot launches (t=8.5 fn=[0,0,0,0], wheels 0.80 in
                # air). Only lift legs whose wheel is still near its terrain.
                # M1seq (2026-08-13): while any wheel swings, the stance axle
                # must stay FIRM and drive. M1reach removed the folded-leg
                # hop (reachable drops), so the body-low clause can fire
                # during the swing again: the rear legs hold the body up
                # while the front lifts (without it, L3 sat under 140N in
                # the force branch and flailed px 0.3/pz +-0.2).
                _pos_lift = (_wz_f > 0.72
                             or float(body['pos'][2])
                             < float(ref.get('z', 0.8)) - 0.06) \
                    and swing[leg] <= 0.5 \
                    and float(wheel_xyz[leg, 2]) \
                    <= float(terrain_h[leg]) + r + 0.05
                # 支撑腿：J^T·(R^T·F_des) 力分配——F_des 是作用在 body 的
                # 接触力（向上），与 VMC 约定一致（f_b=R^T·F，f_sag=[fx,-fz]）
                _qp1 = -1.16 if leg in (0, 1) else 1.16
                _qp2 = 2.30 if leg in (0, 1) else -2.30
                F_w = np.asarray(F_des[leg], dtype=np.float64)
                _hover = False
                if float(wheel_xyz[leg, 2]) > \
                        float(terrain_h[leg]) + r + 0.05:
                    self._hover_t[leg] += dt
                    if self._hover_t[leg] > 0.5:
                        _hover = True
                else:
                    self._hover_t[leg] = 0.0
                if _hover:
                    F_w = np.zeros(3)
                f_b = R.T @ F_w
                fx, fy, fz_up = float(f_b[0]), float(f_b[1]), float(f_b[2])
                # M1fwd (2026-08-13): no BACKWARD force through the legs.
                # The QP realized the nose-up pitch moment with the rear
                # legs' backward F_x (wheel below CoM -> r_z*F_x moment),
                # giving a_x=-10.3 (braking) despite a_des=+11.55 -> the
                # stall at the riser face. Forward traction stays (the
                # drive); backward braking is dropped (the wheels' anti-
                # reverse floor handles the actual braking).
                f_s = np.array([max(fx, 0.0), -fz_up])   # 矢状面 (x, z_down)
                # v1068: 地形阻抗（VMC 同款 kp_h=300）——NMPC F 在过渡态
                # 太小、姿态正则被压过 → 腿泵高发射；阻抗按轮高误差强压
                # M1yyy: 质地阻抗用几何轮心目标（原
                # KIN_TERR 自证 terr=wheel_z-r → _pz_d≈wheel_z
                # → 阻抗/过伸回拉全部失效→
                # riser 棱把轮顶过髋泵高实测）
                _pz_d = float(ref.get('wz_geo', np.zeros(4))[leg]) - float(
                    os.environ.get('S10_NMPC_PRESS', '0.005'))
                _dz_h = _pz_d - float(wheel_xyz[leg, 2])
                _fz_imp = float(os.environ.get(
                    'S10_NMPC_KPH', '300.0')) * _dz_h
                # M1qqq2: 阻抗单向（只压不抬）：
                # 原双向抬升→轮被提前拉起悬空
                # →泵高；单向让轮靠接触滚上立面
                _fz_imp = float(np.minimum(_fz_imp, 0.0))
                f_s = f_s + np.array([0.0, _fz_imp])
                J = self.fk.jac(q1, q2)
                # v1069: 过伸强位置保持——轮高于地形目标时，J^T 在上折位形
                # 失去权威（轮 1.09/body 0.85 实测），改用关节空间高增益
                # PD 把腿拉回标称弯曲位形（防上折，v1014 思路）
                # v2026(m17): 恢复固定蹲姿回拉——几何 IK 在上折位形会选
                # 折叠分支(q1~2.53 钉在限位, J22~0)致 7s 卡死；固定蹲姿
                # 直拉 q1/q2 绕过奇异 J，唯一能掰回前伸位形的路径
                if float(wheel_xyz[leg, 2]) > _pz_d + 0.02:
                    _kpo = 300.0
                    _kdo = 30.0
                    tau[LEG_CTRL_IDX[b + 1]] += _kpo * (_qp1 - q1) - _kdo * dq1
                    tau[LEG_CTRL_IDX[b + 2]] += _kpo * (_qp2 - q2) - _kdo * dq2
                th1, th2 = J.T @ f_s
                # v1056: 姿态正则（零空间 PD 拉回蹲姿，VMC 同款）——力控在
                # 近奇异位形失去权威，腿漂到折叠上伸（轮 1.02/body 0.64
                # 卡死实测）；正则把腿拉回标称蹲姿，防奇异漂移
                _kp_pose = 20.0
                _kd_pose = 6.0
                # v1071: 对侧轴 SWING 时本轴强姿态保持（固定蹲姿）——
                # v1076 IK 姿态版本在真原图更差（东漂+早翻），回退
                # v1100: ???????? SWING ??????????????
                # pz_d???????? body ??????????kp_pose 200 +
                # ???? 300 ??????????real23 ????????
                # 1.1 ???????????? SWING ????????????
                # ?????????v1098 ??????? real29 ?????
                # M1post (2026-08-13): opposite-axle swing -> this axle
                # carries full load (F_z up to 170N). An extended-forward leg
                # maps that force into a knee torque >48Nm -> wheel thrown up
                # (L3 pz -0.3). Strong posture hold (200) beats the
                # force-induced extension; the force then pushes the BODY.
                if float(np.max(swing[2:4])) > 0.5 and leg in (0, 1):
                    _kp_pose = 200.0
                    _kd_pose = 30.0
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
                    _kp_pose = 200.0
                    _kd_pose = 30.0
                tau[LEG_CTRL_IDX[b + 1]] = float(
                    th1 + _kp_pose * (_qp1 - q1) - _kd_pose * dq1)
                tau[LEG_CTRL_IDX[b + 2]] = float(
                    th2 + _kp_pose * (_qp2 - q2) - _kd_pose * dq2)
                # hipx 侧向力（VMC 同款固定 0.30 杠杆）+ 标称外展
                # v1078 回退：HipX 直接 yaw 误差项过冲（t=9 翻），v1075 基线最优
                tau[LEG_CTRL_IDX[b]] = float(0.30 * fy + 80.0 * (
                    -0.05 if leg in (0, 1) else 0.05))
                if _pos_lift:
                    # 位置控制抬 body：目标钳到 0.94（后腿可达 0.96）
                    # + kp 120 防过冲（body 过冲 1.12 > 0.96 后轮折叠实测）
                    # M1reach (2026-08-13): keep the wheel's forward reach
                    # and clamp the drop to the REACHABLE range
                    # sqrt(L^2 - px^2). xd=0 pulled the wheels back (robot
                    # stalled at the riser face); drop 0.30 at px 0.3 is
                    # unreachable (0.42 > leg 0.36) -> saturated bang-bang,
                    # body rise 3.5cm/s too slow. Continuous IK (M1ik)
                    # keeps the target smooth.
                    _zr = min(float(ref.get('z', 0.8)), 0.94)
                    _relx_l = float(np.dot(
                        wheel_xyz[leg, :2] - hip_w[:2], R[:, 0][:2]))
                    _relx_l = float(np.clip(_relx_l, 0.0, 0.26))
                    _drop = float(np.clip(
                        _zr - float(wheel_xyz[leg, 2]), 0.10, 0.30))
                    _reach = float(np.sqrt(max(
                        (self.fk.L1 + self.fk.L2) ** 2
                        - _relx_l ** 2 - 1e-3, 0.03)))
                    _drop = float(min(_drop, _reach - 0.01))
                    _q1t, _q2t = self._ik(
                        _relx_l, -_drop, q1, q2, leg=leg)
                    # M1zzz4: 抬升腿 kd 20->40（衰减 body 过冲发射）
                    tau[LEG_CTRL_IDX[b+1]] = 120.0 * (_q1t - q1) - 60.0 * dq1
                    tau[LEG_CTRL_IDX[b+2]] = 120.0 * (_q2t - q2) - 60.0 * dq2
                    tau[LEG_CTRL_IDX[b]] = 0.0
        # 轮：Pfaffian 驱动——τ 跟随 NMPC 接触力前向分量（受摩擦锥约束），
        # vx PID 只做微调，下限仅防倒转。v1065: 前驱下限-5×4=247N 前向力
        # 作用在轮心（CoM 下方）→ 50Nm 俯仰 → 翘起发射（台架实测）；
        # F_des 前馈的推力受摩擦锥/规划约束，不会发射。
        _dfx = -2.0
        _om_cmd = float(cmd.get('omega', 0.0))
        _wv_des = getattr(self, '_wv_des', None)
        _any_sw = float(np.max(swing)) > 0.5
        fwd_w = R @ np.array([1.0, 0.0, 0.0])
        for leg in range(4):
            wq = float(qvel[WHEEL_QV_IDX[leg]])
            v_wheel = -wq * r
            _sd = -1.0 if leg in (0, 2) else 1.0
            F_w = np.asarray(F_des[leg], dtype=np.float64)
            fx_fb = float(np.dot(F_w, fwd_w))
            # M1qqq: 摆腿轮不吃 F_des 前馈（NMPC 20Hz
            # 与 WBC 200Hz 的 swing 状态不一致时，全支撑
            # 解的后退 F_x 会通过 -r*fx_fb 进入摆腿轮
            # → 剑至滑退实测；摆腿轮只留微驱）
            _on_ground = float(wheel_xyz[leg, 2]) <= \
                float(terrain_h[leg]) + r + 0.02
            if swing[leg] > 0.5 and not _on_ground:
                # swing (lifted): keep forward micro-drive
                _tw = -0.5 * 12.0 * (self._vx_f - v_wheel)
            else:
                _tw = -r * fx_fb
                if _wv_des is not None:
                    # M1qqq: 模拟模式轮速参考不可以后退
                    # （滑退时 NMPC 规划负轮速→后轮制动→加剧滑退）
                    # M1lll: 下限跟进指令速度 0.8*vx_f（原固定 0.3
                    # → 停滞时轮转 1.4 但参考 0.3 →后轮制动实测）
                    # M1vs (2026-08-13): the wheel-speed reference floor uses
                    # the ACTUAL body vx, not the commanded vx_f. At the
                    # riser face the command is 1.2 while the robot moves 0.2
                    # -> the wheels spin at 1.2 (1.0 slip, friction wasted) ->
                    # the drive stalls. Matching the actual speed keeps the
                    # wheels in contact and the drive effective.
                    _vref = max(float(_wv_des[leg]),
                                0.8 * float(body['vx']))
                else:
                    _vref = self._vx_f
                # M1yyy4: 摆动期 stance 轮恢复导航 om 差速（登顶
                # 停滞时 yaw 权威优先于推力；原冻结→
                # yaw 缓慢漂移 1.25→2.87 实测）
                # M1spin (2026-08-13): scale the differential by forward
                # speed — at the stall (vx~0.1, front wheels held in the
                # air before the edge) the saturated om +-0.5 rotated the
                # robot in place (yaw 1.3rad -> spin -> L3 flail). No
                # forward speed, no in-place yaw rotation.
                _vref += _sd * _om_cmd * self.track_half * float(
                    np.clip(body['vx'] / 1.0, 0.0, 1.0))
                # vx PID 微调（参考=NMPC Pfaffian 轮速）
                _tw += -0.3 * 12.0 * (_vref - v_wheel)
                # 防倒转下限（小）
                _tw = min(float(_tw), _dfx)
                # 摆动期冻结轮 yaw 修正（用户原案：
                # 轮失权时专心推力，yaw 交给腿）
                if not _any_sw:
                    _tw += _sd * float(qvel[5]) * 8.0 * self.track_half
                    try:
                        _hdg_e = float(getattr(self.stair, '_hdg_err', 0.0))
                        _ky = 8.0
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
        _q2p = float(np.arccos(c2))
        _q2m = -_q2p
        # M1ik (2026-08-13): pick the branch nearest to the current q2
        # (continuous). The old fixed branch (positive, mirrored negative for
        # rear) jumped 1.5+ rad when the leg was in the other branch ->
        # saturated torque -> wheel whip (L3 q2 -0.14 -> -1.90, t2 -207).
        # Nearest-branch keeps the target continuous; mirror preference
        # (front +, rear -) only breaks ties.
        if leg is not None and leg in (2, 3):
            q2n = _q2m if abs(_q2m - q2) <= abs(_q2p - q2) else _q2p
        else:
            q2n = _q2p if abs(_q2p - q2) <= abs(_q2m - q2) else _q2m
        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        return q1n, q2n

    # ---------------- 主入口 ----------------
    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):
        self._t += dt
        body = self._body_state(qpos, qvel)
        if os.environ.get('S10_NMPC_DEBUG', '0') == '1' and                 int(self._t * 200) % 40 == 0:
            _lg = []
            for _b in range(4):
                _q1 = float(qpos[LEG_Q_IDX[_b * 3 + 1]])
                _q2 = float(qpos[LEG_Q_IDX[_b * 3 + 2]])
                _p = self.fk.wheel_pos(_q1, _q2)
                _J = self.fk.jac(_q1, _q2)
                _lg.append('L%d px=%+.3f pz=%+.3f J21=%+.3f J22=%+.3f' % (
                    _b, _p[0], _p[1], _J[1, 0], _J[1, 1]))
            print('[LEG] ' + ' | '.join(_lg), flush=True)
        fwd = np.array([np.cos(body['yaw']), np.sin(body['yaw'])])
        # 速度低通
        _vt = 0.10
        self._vx_f += (float(cmd.get('vx', 0.0)) - self._vx_f) * min(1.0, dt / _vt)
        # ModeSequence（布尔抬升轮）
        swing = self._mode_sequence(body['pos'], fwd, wheel_xyz)
        # M1qqq: swing 状态变化时强制重解 NMPC，
        # 避免 WBC 用旧 swing 的 F_des（前支撑解后轮
        # F_z 无 95N 下限 → 推力弱滑退实测）
        if not hasattr(self, '_sw_prev') or np.any(swing != self._sw_prev):
            self._last_nmpc_t = -1e9
            self._sw_prev = np.array(swing, dtype=np.float64)
        # 轨迹层参考
        ref = self._ref_traj(body, self._vx_f, float(cmd.get('omega', 0.0)),
                             terrain_h)
        # v1055: z_ref 从接管时实际高度 0.5s 斜坡到目标——初始 body 高于
        # 参考时 QP 想下压（轮不能拉）→ 自由落体过冲振荡（台架实测）
        if not hasattr(self, '_z_ref0'):
            self._z_ref0 = float(body['pos'][2])
            self._z_ref_t0 = self._t
        _zr_ramp = 0.5
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
                body, ref, swing, wheel_xyz, _nmpc_dt, terrain_h)
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
