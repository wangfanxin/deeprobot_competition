# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = (float(qvel[5]) * _kd_y
                                if float(np.max(step_lift[2:4])) > 0.5 else 0.0)
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half"""
assert old in src
new = """                    # v948: 支撑轮 yaw 阻尼差速在任何 swing 期生效——v947
                    # 关掉 QP 腿部 yaw 阻尼后 yaw 漂移(-2.45 实测)；轮子差
                    # 速做 yaw 阻尼（有接触支撑，不会像腿部侧向力那样造成
                    # 载荷不对称）
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = float(qvel[5]) * _kd_y
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v948")