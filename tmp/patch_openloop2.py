# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py")
src = p.read_text(encoding="utf-8-sig")
old = """        try:
            _vx_f = float(getattr(self, "_vx_f", 0.0))
            for _leg in range(4):
                _wq = float(qvel[WHEEL_QV_IDX[_leg]])
                _vw = -_wq * self.fk.r
                _tw = (-(self.wheel_k * (_vx_f - _vw))
                       - self.wheel_d * _wq)
                tau[WHEEL_Q_IDX[_leg]] = float(np.clip(_tw, -13.5, 13.5))
        except Exception:
            pass"""
assert old in src
new = """        try:
            # v926: 爬升期轮矩开环——swing 期间支撑轮满前驱(-13.5)、抬升轮
            # 微正转(-1.5)。速度 PID 在前轮上台面后空转时刹车(+13.5 反向
            # 实测) → 后轮爬顶时前轮没有前向拉力、body 不前进。用户 A3:
            # "前轮贴面期间后轮力矩拉满"，对后轮爬顶即前轮满驱。
            _any_swing = float(np.max(step_lift)) > 0.5
            for _leg in range(4):
                if _any_swing:
                    _slw = float(step_lift[_leg])
                    tau[WHEEL_Q_IDX[_leg]] = -1.5 if _slw > 0.5 else -13.5
                else:
                    _wq = float(qvel[WHEEL_QV_IDX[_leg]])
                    _vw = -_wq * self.fk.r
                    _tw = (-(self.wheel_k * (self._vx_f - _vw))
                           - self.wheel_d * _wq)
                    tau[WHEEL_Q_IDX[_leg]] = float(np.clip(_tw, -13.5, 13.5))
        except Exception:
            pass"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")