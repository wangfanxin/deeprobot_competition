#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        q1n = float(np.clip(q1n, -1.7, 1.0))
        q2n = float(np.clip(q2n, -1.0, 3.0))
        return q1n, q2n"""
new = """        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        # 用物理限位(±2.53/±2.72)——原 [-1.7,1.0] 挡住后腿标称 q1=+1.10
        # → 后腿达不到目标折叠上翻 0.98 实测。分支选择已防镜像，宽限安全。
        q1n = float(np.clip(q1n, -2.5, 2.5))
        q2n = float(np.clip(q2n, -2.7, 3.0))
        return q1n, q2n"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")