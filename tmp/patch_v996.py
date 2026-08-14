#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# re-add face drive block (before wheel loop, after _fd_tau removed... check current)
if "_face_drive" not in src.split("for leg in range(4):")[0]:
    old = """        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        for leg in range(4):"""
    new = """        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        # v996: 贴面区判定(swing 跟随与轮层前驱共用)
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
    assert old in src, "editA"
    src = src.replace(old, new)

# swing: tight follow in face zone (override absolute target)
old = """                # v994/v995: 悬空轮"收腿"——轮高于目标(wheel_z > wz_t+0.01)
                # 即悬空。悬空时腿把轮前推(relx 0.15+)的反作用把 body 往后
                # 拉，后轮推力被抵消、狗死锁 8s(实测 fn=0 + 后轮 6-13.5Nm
                # 前驱推不动)。目标 relx 收到 TUCK_RELX(0.06)，后轮推着
                # body 前进，悬空轮自然漂过棱后落到台面。
                if float(wheel_xyz[leg, 2]) > _wz_t + 0.01:
                    _rel[0] = min(float(_rel[0]), float(os.environ.get(
                        "S10_QP_TUCK_RELX", "0.06")))"""
new = """                # v994/v995: 悬空轮"收腿"——轮高于目标即悬空，目标 relx
                # 收回，消除前推反作用。
                if float(wheel_xyz[leg, 2]) > _wz_t + 0.01:
                    _rel[0] = min(float(_rel[0]), float(os.environ.get(
                        "S10_QP_TUCK_RELX", "0.06")))
                # v996: 贴面紧跟随——目标=轮高+3mm(轻引导)，轮靠前驱滚上
                # 立面；v991 爆炸是 gap 5mm+驱动 8Nm，此版更柔。
                if _face_drive[leg]:
                    _wz_t = float(wheel_xyz[leg, 2]) + float(os.environ.get(
                        "S10_QP_FOLLOW_GAP", "0.003"))"""
assert old in src, "editB"
src = src.replace(old, new)

# face drive usage in wheel loop (check if present)
if "min(float(_tw), _fd_tau)" not in src:
    old2 = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    # v993: 支撑轮前驱下限"""
    new2 = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    if _face_drive[leg]:
                        _tw = min(float(_tw), _fd_tau)
                    # v993: 支撑轮前驱下限"""
    assert old2 in src, "editC"
    src = src.replace(old2, new2)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")