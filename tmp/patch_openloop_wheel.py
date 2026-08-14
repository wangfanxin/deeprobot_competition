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
            # v915: 爬升期轮驱动力开环+限幅（用户改进4）——速度 PID 在前轮
            # 上台面空转时把轮子刹车(+13.5 反向实测)，后轮爬顶无前向推力；
            # 改支撑轮满前驱 -13.5、抬升轮微正转 -1.5 防卡沿
            for _leg in range(4):
                _slw = float(step_lift[_leg])
                _tw = -1.5 if _slw > 0.5 else -13.5
                tau[WHEEL_Q_IDX[_leg]] = float(_tw)
        except Exception:
            pass"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")