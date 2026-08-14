#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        # v1021: 支撑轮前驱下限——爬升期后轮空转被速度 PID 判超速倒转
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
            pass"""
new = """        # v1021: 支撑轮前驱下限——爬升期后轮空转被速度 PID 判超速倒转
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
                    else:
                        _d_fc = 1e9
                        for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                            if _dhv <= 0.085:
                                continue
                            _ddc = float(np.dot(
                                wheel_xyz[_leg, :2] - _rp, _tng))
                            if -0.12 < _ddc < 0.05 and abs(_ddc) < abs(_d_fc):
                                _d_fc = _ddc
                        if _d_fc < 0.9:
                            tau[WHEEL_Q_IDX[_leg]] = min(
                                float(tau[WHEEL_Q_IDX[_leg]]),
                                -float(os.environ.get(
                                    "S10_FP_FACE_DRIVE", "3.0")))
        except Exception:
            pass"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")