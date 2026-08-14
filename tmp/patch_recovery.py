#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）"""
new = """        # 阶段 A：位形恢复（终版批准）——前腿 relx<0（后折）或垂距<3cm
        # （近水平）时，关节空间直接加恢复矩（不经过 J^T，奇异区不被
        # 吃掉），把腿掰回前伸位形；同时后轮满驱推身、前轮微正转。
        # 退出条件：relx>0 且垂距>5cm 持续 0.1s（阶段 B 几何举身接管）。
        try:
            _recov_k = float(os.environ.get("S10_FP_RECOV_K", "15.0"))
            _recov_on = False
            for _leg in (0, 1):
                _hip_r = body["pos"] + body["R"] @ np.array(
                    [0.2277, 0.0, 0.0])
                _relx_r = (np.cos(body["yaw"])
                           * (wheel_xyz[_leg, 0] - _hip_r[0])
                           + np.sin(body["yaw"])
                           * (wheel_xyz[_leg, 1] - _hip_r[1]))
                _drop_r = float(_hip_r[2] - wheel_xyz[_leg, 2])
                if _relx_r < 0.0 or _drop_r < 0.03:
                    _recov_on = True
                    _q1 = float(qpos[6 + _leg * 3 + 1])
                    _tau_r = _recov_k * max(0.05 - _relx_r, 0.0)
                    # 单位换算：relx 米 → 关节力矩，放大到有效幅值
                    _tau_r = float(np.clip(_tau_r * 20.0, -48.0, 48.0))
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r
            if _recov_on:
                for _leg in (2, 3):
                    tau[WHEEL_Q_IDX[_leg]] = -13.5
                for _leg in (0, 1):
                    if float(step_lift[_leg]) <= 0.5:
                        tau[WHEEL_Q_IDX[_leg]] = -3.0
        except Exception:
            pass
        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")