#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）"""
new = """        # 支撑腿防过伸（v1014 温和版）：轮高于几何目标+3cm 时，沿 pz 梯度
        # 方向施加下压力（增益 300、上限 30Nm，配合最终 clip）——后轴
        # SWING 时前轮过伸 0.88-1.2 → roll 崩实测。
        try:
            _geo_o = self._geo_terrain(wheel_xyz)
            for _leg in range(4):
                if step_lift[_leg] > 0.5:
                    continue
                _over = float(wheel_xyz[_leg, 2]) - (
                    float(_geo_o[_leg]) + self.fk.r + 0.03)
                if _over > 0.0:
                    _q1o = float(qpos[6 + _leg * 3 + 1])
                    _q2o = float(qpos[6 + _leg * 3 + 2])
                    _Jo = self.fk.jac(_q1o, _q2o)
                    _Jzo = np.array([_Jo[1, 0], _Jo[1, 1]])
                    _nzo = float(np.linalg.norm(_Jzo)) + 1e-6
                    _Fo = float(np.clip(_over * 300.0, 0.0, 30.0))
                    _to = _Jzo / _nzo * _Fo
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += float(_to[0])
                    tau[LEG_CTRL_IDX[_leg * 3 + 2]] += float(_to[1])
        except Exception:
            pass
        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")