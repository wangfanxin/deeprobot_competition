#!/usr/bin/env python3
import io

# revert body-force in StairWBC (keep it clean)
p2 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
s2 = io.open(p2, "r", encoding="utf-8").read()
old = """        # 直接 body 高度力（打破位置控制"死举"）——body 低于目标时支撑腿
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
new = """        # 腿控 Yaw：HipX 外展/内收修正航向（导航 yaw 误差 + yaw 率阻尼）"""
assert old in s2, "stw revert anchor"
s2 = s2.replace(old, new)
io.open(p2, "w", encoding="utf-8").write(s2)

# base: bdes_z uses raw geometric stance terrain (not closure-collapsed targets)
p1 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py"
s1 = io.open(p1, "r", encoding="utf-8").read()
old = """            # v897: bdes_z 只用支撑腿目标算（前轮 SWING 时高抬升目标不再
            # 把 body 拉高——台架实测 bz 0.92→后轮离地 0.98→yaw 自旋）
            _wz_st = [_wz_all[i] for i in range(4) if _sl_all[i] <= 0.5]"""
new = """            # v897: bdes_z 只用支撑腿目标算（前轮 SWING 时高抬升目标不再
            # 把 body 拉高——台架实测 bz 0.92→后轮离地 0.98→yaw 自旋）
            # v1005: 用原始几何地形(terrain_h+r)而非闭环修正后的 _wz_all——
            # body 低 → wz 目标被闭环压低 → bdes_z 跟着塌 → 狗稳定在
            # 塌陷平衡(body 0.63、腿水平死举，实测)。
            _wz_st = [float(terrain_h[i]) + self.fk.r
                      for i in range(4) if _sl_all[i] <= 0.5]"""
assert old in s1, "base anchor"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)
print("patched OK")