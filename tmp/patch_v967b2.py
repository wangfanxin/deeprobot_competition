#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        qp_stance = stance_mask.copy()
        for _i in range(4):
            if step_lift[_i] > 0.5:
                _lo_z = float(terrain_h[_i]) + self.fk.r + 0.015
                if float(wheel_xyz[_i, 2]) > _lo_z:
                    qp_stance[_i] = 0.0"""
new = """        qp_stance = np.ones(4)
        for _i in range(4):
            if step_lift[_i] > 0.5:
                _lo_z = float(terrain_h[_i]) + self.fk.r + 0.015
                if float(wheel_xyz[_i, 2]) > _lo_z:
                    qp_stance[_i] = 0.0
                # 轮还在地上 → 保持支撑（v967b: 原从 stance_mask 复制导致
                # SWING 轮永远 0，接触感知从未生效——必须在原地轮时置回 1）"""
assert old in src, "anchor missing"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")