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
                    # v990: 差速反馈任意 swing 期启用(原仅后轮爬顶)——前轮
                    # 爬升期无转向能力，贴面力不对称→狗向西漂 2m、d 变负、
                    # 前轮永远无法释放(v989 实测)。轴式调度下对称，安全。
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = float(qvel[5]) * _kd_y
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")