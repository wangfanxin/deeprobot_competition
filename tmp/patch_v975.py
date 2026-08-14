#!/usr/bin/env python3
import io
path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()
old = """                            # v974: q1 上界 -0.35 挡住台面正确解(需 q1≈0)，
                            # 轮悬空 4cm 无法落地；物理限位 ±2.53，放宽到 0.3
                            _q1t = float(np.clip(_q1t, -1.7, 0.3))"""
new = """                            # v974/v975: q1 上界放宽到 1.0——台面支撑解需要
                            # 大腿更水平(q1≈0.7)，0.3 仍顶到边界；物理 ±2.53
                            _q1t = float(np.clip(_q1t, -1.7, 1.0))"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched")