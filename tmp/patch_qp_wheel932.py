# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        for leg in range(4):
            _wq = float(qvel[WHEEL_QV_IDX[leg]])
            _vw = -_wq * self.fk.r
            if stance_mask[leg] > 0.5:
                if _any_swing:
                    # v931: 爬升期支撑轮开环满前驱（用户 A3）——速度 PID 在
                    # 前轮上台面空转时刹车(+13.5 反向)，后轮爬顶时前轮没有
                    # 前向拉力、body 不前进实测
                    tau[WHEEL_Q_IDX[leg]] = -13.5
                else:
                    _tw = -(self.wheel_k * (self._vx_f - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = -1.5"""
assert old in src
new = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _rear_swing = float(np.max(step_lift[2:4])) > 0.5
        for leg in range(4):
            _wq = float(qvel[WHEEL_QV_IDX[leg]])
            _vw = -_wq * self.fk.r
            if stance_mask[leg] > 0.5:
                if _rear_swing:
                    # v932: 仅后轮爬顶期前轮（支撑）开环满前驱——用户 A3：
                    # 前轮在台面上需拉力把车身拉过棱；v931 全爬升期开环在
                    # 前轮爬升期引发自旋（实测回退）
                    tau[WHEEL_Q_IDX[leg]] = -13.5
                else:
                    _tw = -(self.wheel_k * (self._vx_f - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = -1.5"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched qp wheel v932")