#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        # 阶段 A：位形恢复（终版批准）——前腿 relx<0（后折）或垂距<3cm
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
                # v1011: 仅当前腿未 SWING 时触发——SWING 期轮在髋附近
                # (垂距<3cm 正常)，恢复矩误触发 + 后轮前驱 → 弹射
                if (step_lift[_leg] <= 0.5
                        and (_relx_r < 0.0 or _drop_r < 0.03)):
                    _recov_on = True
                    _q1 = float(qpos[6 + _leg * 3 + 1])
                    _tau_r = _recov_k * max(0.05 - _relx_r, 0.0)
                    # 终版公式：K_recov=15 Nm/m * relx 误差，温和恢复
                    _tau_r = float(np.clip(_tau_r, -48.0, 48.0))
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r
            if _recov_on:
                _rdrive = -float(os.environ.get("S10_FP_RECOV_DRIVE", "8.0"))
                for _leg in (2, 3):
                    tau[WHEEL_Q_IDX[_leg]] = _rdrive
                for _leg in (0, 1):
                    if float(step_lift[_leg]) <= 0.5:
                        tau[WHEEL_Q_IDX[_leg]] = -3.0
        except Exception:
            pass"""
new = """        # 阶段 A：位形恢复（终版批准）——折叠腿（relx<0 / 垂距<3cm / 轮高
        # 于几何目标+3cm）时，关节空间直接把 q1 拉回标称位形（不经过
        # J^T，奇异区不被吃掉），让腿获得垂直力臂；配合后轮前驱推身。
        # 关键：折叠位形下力控压轮无效（近奇异），必须先掰回非折叠位形。
        try:
            _recov_k = float(os.environ.get("S10_FP_RECOV_K", "15.0"))
            _recov_kq = float(os.environ.get("S10_FP_RECOV_KQ", "100.0"))
            _recov_on = False
            _geo_r = self._geo_terrain(wheel_xyz)
            for _leg in range(4):
                if step_lift[_leg] > 0.5:
                    continue
                _hip_r = body["pos"] + body["R"] @ np.array(
                    [0.2277 if _leg in (0, 1) else -0.2277, 0.0, 0.0])
                _relx_r = (np.cos(body["yaw"])
                           * (wheel_xyz[_leg, 0] - _hip_r[0])
                           + np.sin(body["yaw"])
                           * (wheel_xyz[_leg, 1] - _hip_r[1]))
                _drop_r = float(_hip_r[2] - wheel_xyz[_leg, 2])
                _folded = (_relx_r < 0.0 or _drop_r < 0.03
                           or float(wheel_xyz[_leg, 2])
                           > float(_geo_r[_leg]) + self.fk.r + 0.03)
                if _folded:
                    _recov_on = True
                    _q1 = float(qpos[6 + _leg * 3 + 1])
                    # q1 标称：前腿 -1.2 / 后腿 +1.2（normal stance）
                    _q1_nom = -1.2 if _leg in (0, 1) else 1.2
                    _tau_r = (_recov_kq * (_q1_nom - _q1)
                              + _recov_k * max(0.05 - _relx_r, 0.0))
                    _tau_r = float(np.clip(_tau_r, -48.0, 48.0))
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r
            if _recov_on:
                _rdrive = -float(os.environ.get("S10_FP_RECOV_DRIVE", "8.0"))
                for _leg in (2, 3):
                    if step_lift[_leg] <= 0.5:
                        tau[WHEEL_Q_IDX[_leg]] = _rdrive
                for _leg in (0, 1):
                    if float(step_lift[_leg]) <= 0.5:
                        tau[WHEEL_Q_IDX[_leg]] = -3.0
        except Exception:
            pass"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")