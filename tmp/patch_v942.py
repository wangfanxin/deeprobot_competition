# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _rear_swing = float(np.max(step_lift[2:4])) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        for leg in range(4):
            _wq = float(qvel[WHEEL_QV_IDX[leg]])
            _vw = -_wq * self.fk.r
            if stance_mask[leg] > 0.5:
                if _rear_swing:
                    # v938: 后轮爬顶期前轮用 yaw 率阻尼差速——v932 开环满
                    # 驱在 2 点支撑期引发 yaw 自旋→roll 崩（实测 yaw 1.66
                    # →2.85）；差速项按 yaw 率反向分配，抵抗自旋，同时保持
                    # 前向驱动
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _vref = (self._vx_f
                             + _side_s[leg] * _kd_y
                             * (-float(qvel[5])) * self.track_half)
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
                else:
                    _tw = -(self.wheel_k * (self._vx_f - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = -1.5"""
assert old in src
new = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        for leg in range(4):
            _wq = float(qvel[WHEEL_QV_IDX[leg]])
            _vw = -_wq * self.fk.r
            if stance_mask[leg] > 0.5:
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
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
                else:
                    _tw = -(self.wheel_k * (self._vx_f - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = -1.5"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v942")