#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# stance branch q2 clip
old = """                            _q1t = float(np.clip(_q1t, -1.7, 1.0))
                            _q2t = float(np.clip(_q2t, -0.2, 3.0))"""
new = """                            _q1t = float(np.clip(_q1t, -1.7, 1.0))
                            # v978: q2 下界放宽到 -1.0——0.5 让短垂距解(轮在
                            # 台面、body 0.83 只需 8cm 垂距)不可达，IK 被迫
                            # 向上折叠(q2≈2.8 轮越过髋到 0.91 悬空实测)
                            _q2t = float(np.clip(_q2t, -1.0, 3.0))"""
assert old in src, "edit1"
src = src.replace(old, new)

# swing branch q2 clip
old = """                    q1t = float(np.clip(q1t, -1.1, 0.2))
                    q2t = float(np.clip(q2t, 0.5, 3.0))"""
new = """                    q1t = float(np.clip(q1t, -1.1, 0.5))
                    # v978: q2 下界 -1.0（同支撑腿）——0.5 强制向上折叠
                    q2t = float(np.clip(q2t, -1.0, 3.0))"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")