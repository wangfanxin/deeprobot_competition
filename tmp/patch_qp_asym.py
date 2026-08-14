# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                _kps = float(os.environ.get("S10_QP_KP_SW", str(self.kp)))
                tau[hipy_i] = (_kps * (q1t - q1)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kps * (q2t - q2)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 2]]))"""
assert old in src
new = """                # v934: 前后轴抬升增益不对称——前轮爬升有动量辅助用软增益
                # （防过伸/泵高）；后轮爬顶需主动抬升 0.125m 用硬增益
                _kps_d = float(os.environ.get("S10_QP_KP_SW", str(self.kp)))
                if leg in (2, 3):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_REAR",
                                                  str(self.kp)))
                tau[hipy_i] = (_kps_d * (q1t - q1)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kps_d * (q2t - q2)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 2]]))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched qp asym")