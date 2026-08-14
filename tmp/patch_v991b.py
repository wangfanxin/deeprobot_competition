#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# Move face_drive computation BEFORE leg loop: insert after self._step_lift_last
old = """        self._step_lift_last = step_lift.copy()"""
new = """        self._step_lift_last = step_lift.copy()
        # v991: 贴面区判定(腿部 swing 跟随与轮层前驱共用)——提前到腿循环前
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
        _fd_tau = -float(os.environ.get("S10_QP_FACE_DRIVE", "8.0"))"""
assert old in src, "edit1"
src = src.replace(old, new)

# Remove the duplicate definition in the wheel loop
old = """        # v991: 贴面区强制前驱——轮滚上立面(0.125m 台阶>轮半径 0.081，纯腿
        # 抬升悬空、纯动量滚不上均实测失败；轮驱+腿跟随是唯一可行机制)。
        # 轴式调度对称爬升，双轮一起滚。
        _face_drive = np.zeros(4, dtype=bool)
        try:
            _fd_lo = float(os.environ.get("S10_QP_FACE_DRIVE_LO", "-0.12"))
            _fd_hi = float(os.environ.get("S10_QP_FACE_DRIVE_HI", "0.05"))
            for _fl in range(4):
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    _dhmin = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
                    if _dhv <= _dhmin:
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
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")