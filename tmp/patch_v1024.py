#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        # v1021/v1023: 支撑轮前驱下限 + yaw_rate 差速——爬升期后轮空转被
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
new = """        # v1021/v1023/v1024: 支撑轮前驱下限(全程)+ yaw_rate 差速——爬升期
        # 后轮空转被 PID 倒转 → 卡死；狗在 StairWBC 接管后前速仅 0.1m/s
        # (速度 PID 驱动不足) → 无法推进到棱口。按终版"轮开环限幅"，
        # StairWBC 全程支撑轮至少 -DRIVE_FLOOR 前驱(负=前)+ yaw_rate 差速。
        try:
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