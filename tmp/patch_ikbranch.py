#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        q2n = float(np.arccos(c2))
        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        q1n = float(np.clip(q1n, -1.7, 1.0))
        q2n = float(np.clip(q2n, -1.0, 3.0))
        return q1n, q2n"""
new = """        q2n = float(np.arccos(c2))
        # 分支按当前 q2 符号：前腿标称 q2>0 用 +acos，后腿标称 q2<0 用
        # -acos（镜像腿）。固定 +acos 会让后腿折叠上翻 1.35+（实测）。
        if q2 < 0:
            q2n = -q2n
        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        q1n = float(np.clip(q1n, -1.7, 1.0))
        q2n = float(np.clip(q2n, -1.0, 3.0))
        return q1n, q2n"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")