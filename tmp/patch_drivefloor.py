#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        # 爬升瞬态：轮矩纯前驱（冻结差速，航交由 HipX 修正）
        try:
            _vx_f = float(getattr(self, "_vx_f", 0.0))
            for _leg in range(4):
                _wq = float(qvel[WHEEL_QV_IDX[_leg]])
                _vw = -_wq * self.fk.r
                _tw = (-(self.wheel_k * (_vx_f - _vw))
                       - self.wheel_d * _wq)
                tau[WHEEL_Q_IDX[_leg]] = float(np.clip(_tw, -13.5, 13.5))
        except Exception:
            pass"""
new = """        # 爬升瞬态：轮矩纯前驱（冻结差速，航交由 HipX 修正）。
        # 前驱下限：狗撞棱 body 停、后轮空转超速被 PID 倒转(tauW=+9/10
        # 实测) → 狗被夹死。SWING 期支撑轮至少 -DRIVE_FLOOR 前驱。
        _df = -6.0
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
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")