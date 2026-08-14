#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        try:
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
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r"""
new = """        try:
            _recov_k = float(os.environ.get("S10_FP_RECOV_K", "15.0"))
            _recov_kq = float(os.environ.get("S10_FP_RECOV_KQ", "200.0"))
            _recov_on = False
            _geo_r = self._geo_terrain(wheel_xyz)
            for _leg in range(4):
                _is_sw = step_lift[_leg] > 0.5
                _hip_r = body["pos"] + body["R"] @ np.array(
                    [0.2277 if _leg in (0, 1) else -0.2277, 0.0, 0.0])
                _relx_r = (np.cos(body["yaw"])
                           * (wheel_xyz[_leg, 0] - _hip_r[0])
                           + np.sin(body["yaw"])
                           * (wheel_xyz[_leg, 1] - _hip_r[1]))
                _drop_r = float(_hip_r[2] - wheel_xyz[_leg, 2])
                # v1020: 折叠判定——支撑腿 relx<0/垂距<3cm；SWING 腿仅当轮
                # 高于几何目标+5cm（真过伸）时强掰 q1 回标称(KQ=200)。
                # 近奇异折叠位形只有先解除折叠、阻抗才有效。
                _over_sw = (_is_sw and float(wheel_xyz[_leg, 2])
                            > float(_geo_r[_leg]) + self.fk.r + 0.05)
                _folded = (not _is_sw and (_relx_r < 0.0 or _drop_r < 0.03)
                           or _over_sw)
                if _folded:
                    _recov_on = True
                    _q1 = float(qpos[6 + _leg * 3 + 1])
                    _q1_nom = -1.2 if _leg in (0, 1) else 1.2
                    _tau_r = (_recov_kq * (_q1_nom - _q1)
                              + _recov_k * max(0.05 - _relx_r, 0.0))
                    _tau_r = float(np.clip(_tau_r, -48.0, 48.0))
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")