#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        # v1021: 支撑轮前驱下限——爬升期后轮空转被速度 PID 判超速倒转
        # → 狗卡在棱口不推进(y 卡 38.0、vx≈0 实测)。SWING 期支撑轮至少
        # -DRIVE_FLOOR 前驱(负=前)。
        # v1022: 贴面前轮温和前驱 -3Nm 滚上立面——狗卡在 d=-0.1(轮贴面
        # 前 3cm)不推进，SWING 目标=地面不抬。body 已健康(0.80)+阻抗已
        # 生效，温和前驱让轮滚上棱(此前爆炸因 body 塌+无阻抗)。
        try:
            _any_swx = float(np.max(step_lift)) > 0.5
            if _any_swx:
                _dfx = -float(os.environ.get("S10_FP_DRIVE_FLOOR", "6.0"))
                for _leg in range(4):
                    if step_lift[_leg] <= 0.5:
                        tau[WHEEL_Q_IDX[_leg]] = min(
                            float(tau[WHEEL_Q_IDX[_leg]]), _dfx)
        except Exception:
            pass"""
new = """        # v1021/v1023: 支撑轮前驱下限 + yaw_rate 差速——爬升期后轮空转被
        # PID 倒转 → 卡死；且前轮贴面不对称让 yaw 漂到 2.1(偏 34°)，前驱
        # 全变西向推力(实测)。SWING 期支撑轮至少 -DRIVE_FLOOR 前驱(负=前)
        # + yaw_rate 差速把航向拉回。
        try:
            _any_swx = float(np.max(step_lift)) > 0.5
            if _any_swx:
                _dfx = -float(os.environ.get("S10_FP_DRIVE_FLOOR", "6.0"))
                _kd_yx = float(os.environ.get("S10_FP_YAW_DIFF", "2.0"))
                for _leg in range(4):
                    if step_lift[_leg] <= 0.5:
                        _sx = -1.0 if _leg in (0, 1) else 1.0
                        _corr = _sx * float(qvel[5]) * _kd_yx * self.track_half
                        tau[WHEEL_Q_IDX[_leg]] = min(
                            float(tau[WHEEL_Q_IDX[_leg]]), _dfx) - _corr
        except Exception:
            pass"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")