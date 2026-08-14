# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py")
src = p.read_text(encoding="utf-8-sig")
old = """            tau[hipx_i] = (self.kp * (self.pose_target[b] - qhx)
                           - self.kd * float(qvel[6 + LEG_QV_LEG[b]]))"""
assert old in src
new = """            # v912: hipx 增益可调 + roll 跟随——固定 ±0.05 目标在爬升
            # 倾斜时被 kp=220 猛拉饱和(±48 振荡 → yaw/roll 乱源实测)；降
            # 增益并随 body roll 外展保持轮贴地
            _kpx_h = float(os.environ.get("S10_FP_KP_HIPX", "60.0"))
            _kdx_h = float(os.environ.get("S10_FP_KD_HIPX", "6.0"))
            _q0h = (self.pose_target[b]
                    - 0.15 * float(np.sign(0.5 - (leg % 2))) * float(body["roll"]))
            tau[hipx_i] = (_kpx_h * (_q0h - qhx)
                           - _kdx_h * float(qvel[6 + LEG_QV_LEG[b]]))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")