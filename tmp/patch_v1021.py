#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）"""
new = """        # v1021: 支撑轮前驱下限——爬升期后轮空转被速度 PID 判超速倒转
        # → 狗卡在棱口不推进(y 卡 38.0、vx≈0 实测)。SWING 期支撑轮至少
        # -DRIVE_FLOOR 前驱(负=前)。
        try:
            _any_swx = float(np.max(step_lift)) > 0.5
            if _any_swx:
                _dfx = -float(os.environ.get("S10_FP_DRIVE_FLOOR", "6.0"))
                for _leg in range(4):
                    if step_lift[_leg] <= 0.5:
                        tau[WHEEL_Q_IDX[_leg]] = min(
                            float(tau[WHEEL_Q_IDX[_leg]]), _dfx)
        except Exception:
            pass
        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")