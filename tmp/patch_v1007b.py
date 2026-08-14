#!/usr/bin/env python3
import io

p1 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py"
s1 = io.open(p1, "r", encoding="utf-8").read()
old = """            # v736: IK 坐标 = yaw 前向投影 + 世界 z 差（稳定可达）
            rel = np.array([_cy * _dw[0] + _sy * _dw[1],
                            -_sy * _dw[0] + _cy * _dw[1],
                            _dw[2]])"""
new = """            # v736: IK 坐标 = yaw 前向投影 + 世界 z 差（稳定可达）
            rel = np.array([_cy * _dw[0] + _sy * _dw[1],
                            -_sy * _dw[0] + _cy * _dw[1],
                            _dw[2]])
            # v1007: relx 钳制为正值（轮至少 5cm 在髋前）——接近段减速 body
            # 前冲让前腿向后折叠(relx -0.18~-0.36 实测)，后向腿近奇异无力
            # 举身 → body 塌 0.63 死举。正常站姿腿必须前伸(relx>0)，
            # 才有垂直力分量把 body 举回。
            rel[0] = max(float(rel[0]), 0.05)"""
assert old in s1, "anchor"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)
print("patched OK")