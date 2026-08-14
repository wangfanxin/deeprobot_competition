#!/usr/bin/env python3
import io

# 1) base: body z/pitch 几何前馈（终版修正 v1006）
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
new = """            # v1006(终版修正): body z/pitch 用几何前后轴轮参考（terrain_h+r，
            # StairWBC 已传几何地形）——不再用 stance-only 目标反推。
            # 原"支撑轮均值+STAND_DROP"在前轮 SWING/后轮 STANCE 混合态取
            # 后轮地面均值 → body 目标偏低 → IK 解水平腿 → 雅可比退化 →
            # 死举(body 塌 0.63、垂距 1cm 实测)。几何参考随前轮上台面
            # 同步抬高 → body 目标显式上抬 → IK 解前屈后伸（有垂直分量）。
            _z_geo = [float(terrain_h[i]) + self.fk.r for i in range(4)]
            _bdes_z = float(np.mean(_z_geo)) + float(os.environ.get(
                'S10_FP_STAND_DROP', '0.26'))
            _bdes_pitch = -float(np.arctan2(
                float(np.mean(_z_geo[0:2])) - float(np.mean(_z_geo[2:4])),
                0.456))"""
assert old in s1, "base anchor"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)

# 2) StairWBC: swing_d 0.30 -> 0.15
p2 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
s2 = io.open(p2, "r", encoding="utf-8").read()
old = """        self.swing_d = 0.30        # 抬升窗：棱前提前抬（靠动量越棱，v899）"""
new = """        self.swing_d = 0.15        # 抬升窗：与 ModeSchedule 触发窗对齐（v1006）
        # 原 0.30 在平地上就开始预拉腿目标 → body 被隐性拖低（审阅修正）"""
assert old in s2, "stw anchor"
s2 = s2.replace(old, new)
io.open(p2, "w", encoding="utf-8").write(s2)
print("patched OK")