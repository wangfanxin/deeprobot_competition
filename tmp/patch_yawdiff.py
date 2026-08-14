#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        _df = -6.0
        try:
            _vx_f = float(getattr(self, "_vx_f", 0.0))
            for _leg in range(4):
                _wq = float(qvel[WHEEL_QV_IDX[_leg]])
                _vw = -_wq * self.fk.r
                _tw = (-(self.wheel_k * (_vx_f - _vw))
                       - self.wheel_d * _wq)
                if float(np.max(step_lift)) > 0.5:
                    _tw = max(float(_tw), _df)
                tau[WHEEL_Q_IDX[_leg]] = float(np.clip(_tw, -13.5, 13.5))
        except Exception:
            pass"""
new = """        _df = -6.0
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        try:
            _vx_f = float(getattr(self, "_vx_f", 0.0))
            _any_sw = float(np.max(step_lift)) > 0.5
            for _leg in range(4):
                _wq = float(qvel[WHEEL_QV_IDX[_leg]])
                _vw = -_wq * self.fk.r
                _vref = _vx_f
                if _any_sw:
                    # 前轮抬空后失去 yaw 阻力，支撑轮差速主动抗旋（yaw_rate
                    # 反馈）；同时保持前驱下限防后轮空转倒转
                    _vref = _vx_f - _side_s[_leg] * float(qvel[5]) * 2.0 \
                        * self.track_half
                _tw = (-(self.wheel_k * (_vref - _vw))
                       - self.wheel_d * _wq)
                if _any_sw:
                    _tw = max(float(_tw), _df)
                tau[WHEEL_Q_IDX[_leg]] = float(np.clip(_tw, -13.5, 13.5))
        except Exception:
            pass"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")