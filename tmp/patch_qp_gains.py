# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")

# hipx 增益环境可调
old = "            tau[hipx_i] = self.kp * (_q0_tgt - qhx) - self.kd * float(qvel[6 + LEG_QV_LEG[b]])"
assert old in src
new = "            _kpx = float(os.environ.get(\"S10_QP_KP_HIPX\", str(self.kp)))\n            tau[hipx_i] = _kpx * (_q0_tgt - qhx) - self.kd * float(qvel[6 + LEG_QV_LEG[b]])"
src = src.replace(old, new, 1)

# SWING IK PD 增益环境可调
old2 = """                tau[hipy_i] = (self.kp * (q1t - q1)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (self.kp * (q2t - q2)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 2]]))"""
assert old2 in src
new2 = """                _kps = float(os.environ.get("S10_QP_KP_SW", str(self.kp)))
                tau[hipy_i] = (_kps * (q1t - q1)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kps * (q2t - q2)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 2]]))"""
src = src.replace(old2, new2, 1)

p.write_text(src, encoding="utf-8")
print("patched")