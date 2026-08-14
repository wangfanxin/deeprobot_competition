#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                _drop_r = float(_hip_r[2] - wheel_xyz[_leg, 2])
                if _relx_r < 0.0 or _drop_r < 0.03:"""
new = """                _drop_r = float(_hip_r[2] - wheel_xyz[_leg, 2])
                # v1011: 仅当前腿未 SWING 时触发——SWING 期轮在髋附近
                # (垂距<3cm 正常)，恢复矩误触发 + 后轮前驱 → 弹射
                if (step_lift[_leg] <= 0.5
                        and (_relx_r < 0.0 or _drop_r < 0.03)):"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")