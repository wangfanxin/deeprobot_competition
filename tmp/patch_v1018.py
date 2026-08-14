#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """            for _leg in range(4):
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
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r"""
new = """            for _leg in range(4):
                _is_sw = step_lift[_leg] > 0.5
                if _is_sw and float(wheel_xyz[_leg, 2]) <= (
                        float(_geo_r[_leg]) + self.fk.r + 0.05):
                    continue   # SWING 轮未过伸：保持（避免误触发）
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
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")