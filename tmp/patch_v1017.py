#!/usr/bin/env python3
import io

p1 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py"
s1 = io.open(p1, "r", encoding="utf-8").read()
old = """                pz_des9 = float(terrain_h[leg]) + self.fk.r - _fp_press
                _dz9 = pz_des9 - float(wheel_xyz[leg, 2])
                _F9 = float(os.environ.get('S10_FP_KPH', '300')) * min(_dz9, 0.0)
                _F9 = max(_F9, 2.0)"""
new = """                pz_des9 = float(terrain_h[leg]) + self.fk.r - _fp_press
                _dz9 = pz_des9 - float(wheel_xyz[leg, 2])
                # v1017: 去掉 max(2.0)——原把"轮高于目标"的负下压力钳成
                # +2，折叠过伸完全无对抗（轮 1.2 vs 目标 0.747 实测）。
                # F9<0 = 下压（fz=-f_b9[2]=-F9>0），轮高于目标时强压回位。
                _F9 = float(os.environ.get('S10_FP_KPH', '300')) * min(_dz9, 0.0)
                if _F9 > 0:
                    _F9 = max(_F9, 2.0)"""
assert old in s1, "anchor"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)
print("patched OK")