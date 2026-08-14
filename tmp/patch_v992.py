#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                if _any_swing:
                    # v942: 任何 swing 期支撑轮速度 PID（v958 开环满驱在前
                    # 轮爬升期自旋 3s 翻车回退）
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = (float(qvel[5]) * _kd_y
                                if float(np.max(step_lift[2:4])) > 0.5 else 0.0)
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half"""
new = """                if _any_swing:
                    # v942: 任何 swing 期支撑轮速度 PID（v958 开环满驱在前
                    # 轮爬升期自旋 3s 翻车回退）
                    # v992: 差速反馈任意 swing 期启用——前轮抬空后失去 yaw
                    # 阻力，后轮任何不对称都会让狗转圈(v991 宽窗+动量配置
                    # yaw 1.3→-0.11 自旋翻车实测)；支撑轮差速主动抗旋。
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = float(qvel[5]) * _kd_y
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")