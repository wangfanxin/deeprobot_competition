#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                _tw = (-(self.wheel_k * (_vref - _vw))
                       - self.wheel_d * _wq)
                if _any_sw:
                    _tw = max(float(_tw), _df)
                tau[WHEEL_Q_IDX[_leg]] = float(np.clip(_tw, -13.5, 13.5))"""
new = """                _tw = (-(self.wheel_k * (_vref - _vw))
                       - self.wheel_d * _wq)
                if _any_sw:
                    _tw = max(float(_tw), _df)
                    if step_lift[_leg] > 0.5:
                        # 抬升轮：温和前驱 -3Nm 贴面滚动辅助（位置控制
                        # 约束不爆炸；0 驱动力则轮贴面卡死）
                        _tw = min(float(_tw), -3.0)
                tau[WHEEL_Q_IDX[_leg]] = float(np.clip(_tw, -13.5, 13.5))"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")