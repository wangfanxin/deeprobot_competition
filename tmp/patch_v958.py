# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            if stance_mask[leg] > 0.5:
                if _any_swing:
                    # v942: 任何 swing 期支撑轮开环满前驱（用户 A3）——
                    # 速度 PID 在爬升期把支撑轮刹车(+13.5 反向实测) → 狗
                    # 卡在棱前不前进、后轮够不到 SWING 窗。v931 失败时
                    # 是宽抬升窗+泵高环境，现 v939 窄窗+v935 阻尼已不同。
                    # 后轮爬顶期叠加 yaw 率阻尼差速抗自旋（v938）。
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = (float(qvel[5]) * _kd_y
                                if float(np.max(step_lift[2:4])) > 0.5 else 0.0)
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))"""
assert old in src
new = """            if stance_mask[leg] > 0.5:
                if _any_swing:
                    # v958: 单轮序列(3点支撑)下 swing 期支撑轮开环满前驱
                    # ——速度 PID 爬升期刹车(+13.5 反向)致狗不前进、后轮
                    # 够不到 SWING 窗（3 轮上台面后 RR 窗 d=-0.37 卡死实
                    # 测）。3 点支撑够稳，开环安全；yaw 阻尼仍用差速。
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _vref = self._vx_f
                    _tw = -13.5
                    tau[WHEEL_Q_IDX[leg]] = float(_tw)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v958")