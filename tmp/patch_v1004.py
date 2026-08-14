#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                    if _face_drive[leg] and float(wheel_xyz[leg, 2]) < 0.72:
                        # v998: 轮低于 0.72(贴面爬升中)才前驱，过顶即停——
                        # 前驱持续把轮顶到 1.2+ 过伸(v997 实测)
                        _tw = min(float(_tw), _fd_tau)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))"""
new = """                    if _face_drive[leg] and float(wheel_xyz[leg, 2]) < 0.72:
                        # v998: 轮低于 0.72(贴面爬升中)才前驱，过顶即停——
                        # 前驱持续把轮顶到 1.2+ 过伸(v997 实测)
                        _tw = min(float(_tw), _fd_tau)
                    # v1004: 支撑轮前驱下限——爬升中后轮空转被 PID 判超速
                    # 全力倒转(tauW=+13.5 实测)，狗失去前驱+倒转力矩自旋。
                    # 至少保持 -DRIVE_FLOOR 前驱。
                    _df = -float(os.environ.get("S10_QP_DRIVE_FLOOR", "6.0"))
                    _tw = max(float(_tw), _df)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")