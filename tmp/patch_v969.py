#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# ---------- Edit 1: revert face drive block ----------
old = """        # v968: 贴面区强制前驱——轮在棱口(d∈[-0.12,0.05])贴面时，空转被
        # 速度 PID 判为超速反刹 → 轮顶死在棱上不爬升(实测 d 卡 -0.07 数秒)。
        # 贴面轮至少给 -FACE_DRIVE 前驱(滚上立面)，取 min(前驱, PID)。
        _face_drive = np.zeros(4, dtype=bool)
        try:
            _fd_lo = float(os.environ.get("S10_QP_FACE_DRIVE_LO", "-0.12"))
            _fd_hi = float(os.environ.get("S10_QP_FACE_DRIVE_HI", "0.05"))
            for _fl in range(4):
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    if _dhv <= 0.085:
                        continue
                    _ddw = float(np.dot(wheel_xyz[_fl, :2] - _rp, _tng))
                    if _fd_lo < _ddw < _fd_hi:
                        _face_drive[_fl] = True
                        break
        except Exception:
            pass
        _fd_tau = -float(os.environ.get("S10_QP_FACE_DRIVE", "8.0"))
        for leg in range(4):"""
new = """        for leg in range(4):"""
assert old in src, "edit1 anchor missing"
src = src.replace(old, new)

# ---------- Edit 2: revert face drive usage ----------
old = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    if _face_drive[leg]:
                        _tw = min(float(_tw), _fd_tau)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
                else:
                    _tw = -(self.wheel_k * (self._vx_f - _vw)) - self.wheel_d * _wq
                    if _face_drive[leg]:
                        _tw = min(float(_tw), _fd_tau)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = (_fd_tau if _face_drive[leg]
                                         else -1.5)"""
new = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
                else:
                    _tw = -(self.wheel_k * (self._vx_f - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = -1.5"""
assert old in src, "edit2 anchor missing"
src = src.replace(old, new)

# ---------- Edit 3: attitude control active during ANY swing ----------
old = """            _rear_sw = float(getattr(self, '_rear_swing', 0.0)) > 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0")) if _rear_sw else 0.0
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0")) if _rear_sw else 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]"""
new = """            _rear_sw = float(getattr(self, '_rear_swing', 0.0)) > 0.5
            # v969: roll/pitch 修正任意 SWING 期启用(原仅后轮爬顶)——前轮
            # 单轮抬升时 body 无俯仰控制 → 后仰 0.8-1.0rad 实测；现在有接触
            # 感知支撑+重心解 λ_ref+抬升反作用力矩，QP 有可行域做修正
            _any_sw_q = float(np.min(stance_mask)) < 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0"))
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0"))
            if not _any_sw_q:
                _ar_k = 0.0; _ap_k = 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]"""
assert old in src, "edit3 anchor missing"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")