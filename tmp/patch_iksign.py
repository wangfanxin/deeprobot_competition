#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        L1, L2 = self.fk.L1, self.fk.L2
        r2 = min(xd * xd + zd * zd, (L1 + L2) ** 2 - 1e-6)
        c2 = float(np.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2),
                           -1.0, 1.0))
        q2n = float(np.arccos(c2))
        q1n = float(np.arctan2(xd, zd) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))"""
new = """        L1, L2 = self.fk.L1, self.fk.L2
        zd_d = -zd   # 基类 zd 向上为正 -> 闭式用向下为正
        r2 = min(xd * xd + zd_d * zd_d, (L1 + L2) ** 2 - 1e-6)
        c2 = float(np.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2),
                           -1.0, 1.0))
        q2n = float(np.arccos(c2))
        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")