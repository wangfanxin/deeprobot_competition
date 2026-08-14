# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                    except Exception:
                        th += (_kpp * (self.pose_target[b + 1] - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (self.pose_target[b + 2] - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                else:
                    th += (_kpp * (self.pose_target[b + 1] - q1)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                    tk += (_kpp * (self.pose_target[b + 2] - q2)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                tau[hipy_i] = float(np.clip(th, -48, 48))
                tau[knee_i] = float(np.clip(tk, -48, 48))"""
assert old in src
new = """                    except Exception:
                        th += (_kpp * (self.pose_target[b + 1] - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (self.pose_target[b + 2] - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                else:
                    th += (_kpp * (self.pose_target[b + 1] - q1)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                    tk += (_kpp * (self.pose_target[b + 2] - q2)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                # v955: 支撑腿防过伸——轮在台面上时实际高度超过 geo-top+r
                # 过多(>0.03)就伸膝压回台面（FR 姿态期持续 1.03-1.06 悬空
                # 实测，stance PD 被 body roll 带高拉不回）
                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    if _over_s > 0.03:
                        _k_ovs = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                        tk -= _k_ovs * _over_s
                tau[hipy_i] = float(np.clip(th, -48, 48))
                tau[knee_i] = float(np.clip(tk, -48, 48))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v955")