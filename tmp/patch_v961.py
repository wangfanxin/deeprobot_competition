#!/usr/bin/env python3
import io, re

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# ---------- Edit 1: __init__ lazily tracked swing targets ----------
old1 = """        self._vx_f = 0.0
        self._om_f = 0.0"""
new1 = """        self._vx_f = 0.0
        self._om_f = 0.0
        # v961: 抬升目标轮高（外力矩建模用）+ 抬升目标速率（预留）
        self._sw_z_tgt = None
        self._sw_zt = None"""
assert old1 in src, "edit1 anchor missing"
src = src.replace(old1, new1)

# ---------- Edit 2: _qp_solve attitude block + swing external wrench ----------
old2 = """            _rear_sw = float(getattr(self, '_rear_swing', 0.0)) > 0.5
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
            a_des[5] = _ay_k * getattr(self, '_yaw_rate', 0.0)"""
new2 = """            # v961: 姿态修正(roll/pitch/yaw)在任意 SWING 期激活——原仅后轮
            # 爬顶期启用，前轮爬升期 pitch 无控制 → body 后仰 0.8-1.0 rad
            # 实测(v959 日志 pitch=-0.83..-1.06)。配合抬升腿外力矩建模让 QP
            # 在支撑可行域内主动分配对侧载荷。
            _any_sw_q = float(np.min(stance_mask)) < 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0"))
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0"))
            _ay_k = float(os.environ.get("S10_QP_AY_K", "-20.0"))
            if not _any_sw_q:
                _ar_k = 0.0; _ap_k = 0.0; _ay_k = 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]
            a_des[5] = _ay_k * getattr(self, '_yaw_rate', 0.0)
            # v961: 抬升腿外力矩建模——SWING 腿把轮往上拉，反作用(向下)压
            # 在对应髋关节。把这个已知外力矩加进 b0，QP 才能把姿态修正落在
            # 载荷分配上(否则修正被外力矩吃掉 → 前轮爬升 body 后仰/侧倾)。
            # 大小 ∝ 抬升误差(目标-实际轮高)，方向沿世界 y(roll)/x(pitch)。
            _sw_ff = float(os.environ.get("S10_QP_SW_FF", "1.0"))
            if _sw_ff > 0.0 and _any_sw_q:
                _n_sw = int(np.sum(np.asarray(stance_mask) <= 0.5))
                if _n_sw == 1:
                    _swi = int(np.argmax(np.asarray(stance_mask) <= 0.5))
                    _zt_sw = 0.0
                    if getattr(self, '_sw_z_tgt', None) is not None:
                        _zt_sw = float(self._sw_z_tgt[_swi])
                    _za_sw = float(wheel_xyz[_swi, 2])
                    _err_sw = max(0.0, _zt_sw - _za_sw)
                    if _err_sw > 0.0:
                        _kps_use = float(os.environ.get("S10_QP_KP_SW", "100.0"))
                        if _swi in (2, 3):
                            _kps_use = float(os.environ.get(
                                "S10_QP_KP_SW_REAR", "100.0"))
                        _f_sw = _sw_ff * _kps_use * _err_sw
                        _sx = 0.2277 if _swi in (0, 1) else -0.2277
                        _sy = 0.181 if _swi in (0, 2) else -0.181
                        _hip_sw = body["pos"] + body["R"] @ np.array(
                            [_sx, _sy, 0.0])
                        _rcs = _hip_sw - body["pos"]
                        _ixx = float(self.I_body[0, 0])
                        _iyy = float(self.I_body[1, 1])
                        # M = rc x (0,0,-f): Mx=-rc_y*f, My=+rc_x*f
                        b0[3] += (-_rcs[1] * _f_sw) / _ixx
                        b0[4] += (_rcs[0] * _f_sw) / _iyy"""
assert old2 in src, "edit2 anchor missing"
src = src.replace(old2, new2)

# ---------- Edit 3: λ_ref tripod static solution (single-wheel swing) ----------
old3 = """        # λ_ref：支撑 mg/4 均载，抬升 0
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        for i in range(4):
            if stance_mask[i] > 0.5:
                lam_ref[i, 2] = self.m * self.g / 4.0"""
new3 = """        # λ_ref：支撑载荷基准——单轮抬升时用三脚支撑静解(Σλ=mg，绕 x 力
        # 矩平衡把抬升角份额当作外载)，其余情况 mg/4 均载。QP 的正则项把
        # 解往这个基准拉，配合 b0 外力矩让姿态修正不饱和。
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        _sw_l = [i for i in range(4) if stance_mask[i] <= 0.5]
        if len(_sw_l) == 1:
            _sw_i = _sw_l[0]
            _st_l = [i for i in range(4) if stance_mask[i] > 0.5]
            _yh = [0.181, -0.181, 0.181, -0.181]
            _Aq = np.array([[1.0, 1.0, 1.0],
                            [_yh[_st_l[0]], _yh[_st_l[1]], _yh[_st_l[2]]]])
            _bq = np.array([self.m * self.g,
                            -_yh[_sw_i] * self.m * self.g / 4.0])
            _uq = np.full(3, self.m * self.g / 3.0)
            # 最小范数解 λ = u + A^T (A A^T)^-1 (b - A u)
            _lamq = _uq + _Aq.T @ np.linalg.solve(
                _Aq @ _Aq.T, _bq - _Aq @ _uq)
            for _k, _i in enumerate(_st_l):
                lam_ref[_i, 2] = float(max(_lamq[_k], 10.0))
        else:
            for i in range(4):
                if stance_mask[i] > 0.5:
                    lam_ref[i, 2] = self.m * self.g / 4.0"""
assert old3 in src, "edit3 anchor missing"
src = src.replace(old3, new3)

# ---------- Edit 4: store swing target wheel z (for wrench) ----------
old4 = """                _wz_t = float(terrain_h[leg]) + self.fk.r
                if _pz > 0.01 and _sl > 0.02:
                    # v950: swing 目标严格封顶到台面顶+r+0.005——原
                    # body_z+0.25 松上限允许轮抬到 1.08（FR/RL 过伸实测）；
                    # 台面顶来自 place_z 反解（pz = 顶-r-margin）
                    _wz_t = min(_pz + self.fk.r + 0.04,
                                _pz + self.fk.r + 0.045)"""
new4 = """                _wz_t = float(terrain_h[leg]) + self.fk.r
                if _pz > 0.01 and _sl > 0.02:
                    # v950: swing 目标严格封顶到台面顶+r+0.005——原
                    # body_z+0.25 松上限允许轮抬到 1.08（FR/RL 过伸实测）；
                    # 台面顶来自 place_z 反解（pz = 顶-r-margin）
                    _wz_t = min(_pz + self.fk.r + 0.04,
                                _pz + self.fk.r + 0.045)
                # v961: 记录抬升目标轮高（QP 外力矩建模用）
                if getattr(self, '_sw_z_tgt', None) is None:
                    self._sw_z_tgt = np.zeros(4)
                    self._sw_zt = np.zeros(4)
                self._sw_z_tgt[leg] = _wz_t"""
assert old4 in src, "edit4 anchor missing"
src = src.replace(old4, new4)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK, new length:", len(src.splitlines()), "lines")