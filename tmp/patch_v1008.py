#!/usr/bin/env python3
import io

p2 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
s2 = io.open(p2, "r", encoding="utf-8").read()
old = """        self.swing_d = 0.30        # 抬升窗：棱前提前抬（靠动量越棱，v899）"""
new = """        self.swing_d = 0.15        # 抬升窗：与 ModeSchedule 触发窗对齐（v1008）
        # 0.30 提前预拉腿目标 → body 塌陷(0.63 实测)；0.15 保持 body 0.77"""
assert old in s2, "anchor"
s2 = s2.replace(old, new)
io.open(p2, "w", encoding="utf-8").write(s2)
print("patched")