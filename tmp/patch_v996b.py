#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# 1) insert face_drive right after _step_lift_last
old = """        self._step_lift_last = step_lift.copy()"""
new = """        self._step_lift_last = step_lift.copy()
        # v996: 贴面区判定(swing 跟随与轮层前驱共用)——腿循环前定义
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
        _fd_tau = -float(os.environ.get("S10_QP_FACE_DRIVE", "4.0"))"""
assert old in src, "edit1"
src = src.replace(old, new)

# 2) remove the duplicate in the wheel-loop section
old = """        # v996: 贴面区判定(swing 跟随与轮层前驱共用)
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
        for leg in range(4):"""
new = """        for leg in range(4):"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")