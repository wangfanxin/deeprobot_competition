#!/usr/bin/env python3
import io

# 1) base: expose bdes_z
p1 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py"
s1 = io.open(p1, "r", encoding="utf-8").read()
old = """            _bdes_z = float(np.mean(_wz_st)) + float(os.environ.get(
                'S10_FP_STAND_DROP', '0.26'))"""
new = """            _bdes_z = float(np.mean(_wz_st)) + float(os.environ.get(
                'S10_FP_STAND_DROP', '0.26'))
            self._bdes_z_last = _bdes_z"""
assert old in s1, "base anchor"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)

# 2) StairWBC: direct body-lift force
p2 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
s2 = io.open(p2, "r", encoding="utf-8").read()
old = """        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）"""
new = """        # 直接 body 高度力（打破位置控制"死举"）——body 低于目标时支撑腿
        # 施加向下轮力（J^T 投影），反作用把 body 举起。位置控制靠轮目标
        # 间接抬 body，在腿近乎水平时力是水平的（轮前滚），body 塌 0.63
        # 无法回升（实测）。
        try:
            _bz_des = float(getattr(self, "_bdes_z_last", 0.0))
            _bz_act = float(body["pos"][2])
            if _bz_des > 0.4 and _bz_act < _bz_des - 0.03:
                _fz = min(float(os.environ.get("S10_FP_BODY_FORCE", "150.0"))
                          * (_bz_des - _bz_act), 220.0)
                for _leg in range(4):
                    if step_lift[_leg] > 0.5:
                        continue
                    _q1 = float(qpos[6 + _leg * 3 + 1])
                    _q2 = float(qpos[6 + _leg * 3 + 2])
                    _Jb = self.fk.jac(_q1, _q2)
                    _tb = _Jb.T @ np.array([0.0, _fz])
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += float(_tb[0])
                    tau[LEG_CTRL_IDX[_leg * 3 + 2]] += float(_tb[1])
        except Exception:
            pass
        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）"""
assert old in s2, "stw anchor"
s2 = s2.replace(old, new)
io.open(p2, "w", encoding="utf-8").write(s2)
print("patched OK")