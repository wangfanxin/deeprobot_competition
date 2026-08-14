#!/usr/bin/env python3
import io

p1 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py"
s1 = io.open(p1, "r", encoding="utf-8").read()
old = """            # v897: bdes_z 只用支撑腿目标算（前轮 SWING 时高抬升目标不再
            # 把 body 拉高——台架实测 bz 0.92→后轮离地 0.98→yaw 自旋）
            _wz_st = [_wz_all[i] for i in range(4) if _sl_all[i] <= 0.5]
            if not _wz_st:
                _wz_st = list(_wz_all)
            _bdes_z = float(np.mean(_wz_st)) + float(os.environ.get(
                'S10_FP_STAND_DROP', '0.26'))
            _wz_fm = float(np.mean(_wz_all[0:2]))
            _wz_rm = float(np.mean(_wz_all[2:4]))
            _bdes_pitch = -float(np.arctan2(_wz_fm - _wz_rm, 0.456))"""
new = """            # v1006+v1007(组合): body z/pitch 用几何前后轴轮参考（terrain_h+r，
            # StairWBC 已传几何地形）——配合 relx 钳制，前轮 SWING 时 body
            # 目标随前轮参考显式上抬，IK 解前屈后伸（有垂直分量），打破
            # "水平腿死举"。原 stance-only 均值在混合态取后轮地面 → 偏低。
            _z_geo = [float(terrain_h[i]) + self.fk.r for i in range(4)]
            _bdes_z = float(np.mean(_z_geo)) + float(os.environ.get(
                'S10_FP_STAND_DROP', '0.26'))
            _bdes_pitch = -float(np.arctan2(
                float(np.mean(_z_geo[0:2])) - float(np.mean(_z_geo[2:4])),
                0.456))"""
assert old in s1, "anchor"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)
print("patched OK")