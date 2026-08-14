# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                # v935: swing kd 默认 30（原 8 阻尼比 ~0.25 欠阻尼，轮离地
                # 后自由过冲到 1.1 悬空实测）
                _kds_d = float(os.environ.get("S10_QP_KD_SW", "30.0"))
                tau[hipy_i] = (_kps_d * (q1t - q1)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kps_d * (q2t - q2)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 2]]))"""
assert old in src
new = """                # v935: swing kd 默认 30（原 8 阻尼比 ~0.25 欠阻尼，轮离地
                # 后自由过冲到 1.1 悬空实测）
                _kds_d = float(os.environ.get("S10_QP_KD_SW", "30.0"))
                tau[hipy_i] = (_kps_d * (q1t - q1)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kps_d * (q2t - q2)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                # v952: 防过伸反馈——swing 轮实际高度超过目标>0.02m 时，
                # 膝盖加强制伸展力矩把轮压回台面（过伸是动力学折叠惯性，
                # 位置 PD 拉不回：单轮序列 FR 冲到 1.08 实测）。减少 q2
                # (伸膝) 推轮向下。
                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                if _over2 > 0.02:
                    _k_ov = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                    tau[knee_i] -= _k_ov * _over2"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v952")