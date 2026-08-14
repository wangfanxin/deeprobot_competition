#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# face drive only when wheel below the face-rolling height (anti-overshoot)
old = """                    if _face_drive[leg]:
                        _tw = min(float(_tw), _fd_tau)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))"""
new = """                    if _face_drive[leg] and float(wheel_xyz[leg, 2]) < 0.72:
                        # v998: 轮低于 0.72(贴面爬升中)才前驱，过顶即停——
                        # 前驱持续把轮顶到 1.2+ 过伸(v997 实测)
                        _tw = min(float(_tw), _fd_tau)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))"""
assert old in src, "edit1"
src = src.replace(old, new)

# swing wheel face drive: also gate
old = """            else:
                tau[WHEEL_Q_IDX[leg]] = (_fd_tau if _face_drive[leg]
                                         else -1.5)"""
new = """            else:
                tau[WHEEL_Q_IDX[leg]] = (_fd_tau if (_face_drive[leg]
                                         and float(wheel_xyz[leg, 2]) < 0.72)
                                         else -1.5)"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")