#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# ---------- Edit 1: λ_ref barycentric + parametrized weight ----------
old = """        # λ_ref：支撑 mg/4 均载，抬升 0
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        for i in range(4):
            if stance_mask[i] > 0.5:
                lam_ref[i, 2] = self.m * self.g / 4.0"""
new = """        # λ_ref：支撑载荷基准——单轮抬升时用支撑三角形重心解(CoM 在三角
        # 形内的静态分配)，其余 mg/4 均载。3 点支撑下均载是错误目标：
        # RR 抬起时静解 FR≈RL≈mg/2、FL≈0，均载会让 QP 的载荷分布远离
        # 物理可行域(配合 b0 外力矩让姿态修正不饱和)。
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        _sw_l = [i for i in range(4) if stance_mask[i] <= 0.5]
        if len(_sw_l) == 1:
            _st_l = [i for i in range(4) if stance_mask[i] > 0.5]
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
                if stance_mask[i] > 0.5:
                    lam_ref[i, 2] = self.m * self.g / 4.0"""
assert old in src, "edit1 anchor missing"
src = src.replace(old, new)

# ---------- Edit 2: parametrized λ_ref regularization weight ----------
old = """            P = A.T @ W1 @ A + 1e-2 * np.eye(n)
            q = -A.T @ W1 @ (a_des - b0) - 1e-2 * lam_ref.reshape(-1)"""
new = """            _lamw = float(os.environ.get("S10_QP_LAM_W", "0.05"))
            P = A.T @ W1 @ A + _lamw * np.eye(n)
            q = -A.T @ W1 @ (a_des - b0) - _lamw * lam_ref.reshape(-1)"""
assert old in src, "edit2 anchor missing"
src = src.replace(old, new)

# ---------- Edit 3: b0 swing-leg reaction wrench (dynamic, any swing) ----------
old = """            b0 = np.zeros(6)
            b0[2] = -self.g
            # a_des：z 高度阻尼 + roll/pitch 回零（世界系角加速度近似）"""
new = """            b0 = np.zeros(6)
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
            # a_des：z 高度阻尼 + roll/pitch 回零（世界系角加速度近似）"""
assert old in src, "edit3 anchor missing"
src = src.replace(old, new)

# ---------- Edit 4: store swing target z ----------
old = """                _wz_t = float(terrain_h[leg]) + self.fk.r
                if _pz > 0.01 and _sl > 0.02:
                    # v950: swing 目标严格封顶到台面顶+r+0.005——原
                    # body_z+0.25 松上限允许轮抬到 1.08（FR/RL 过伸实测）；
                    # 台面顶来自 place_z 反解（pz = 顶-r-margin）
                    _wz_t = min(_pz + self.fk.r + 0.04,
                                _pz + self.fk.r + 0.045)"""
new = """                _wz_t = float(terrain_h[leg]) + self.fk.r
                if _pz > 0.01 and _sl > 0.02:
                    # v950: swing 目标严格封顶到台面顶+r+0.005——原
                    # body_z+0.25 松上限允许轮抬到 1.08（FR/RL 过伸实测）；
                    # 台面顶来自 place_z 反解（pz = 顶-r-margin）
                    _wz_t = min(_pz + self.fk.r + 0.04,
                                _pz + self.fk.r + 0.045)
                # v964: 记录抬升目标轮高（QP 外力矩建模用）
                if getattr(self, '_sw_z_tgt', None) is None:
                    self._sw_z_tgt = np.zeros(4)
                self._sw_z_tgt[leg] = _wz_t"""
assert old in src, "edit4 anchor missing"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")