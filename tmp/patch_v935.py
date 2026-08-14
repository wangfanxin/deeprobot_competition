# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
# 1) 恢复释放阈值 0.10（滞回：触发<0.05, 释放>0.10）
old = """            # v930: 前轮释放阈值 0.10→0.05——d∈(0.05,0.10) 带里前轮反复
            # 重触发阻塞后轮 SWING（QP 台架 y38.6 前轮上台面但后轮不爬
            # 实测）；放宽后前轮早释放、后轮可触发
            if _df > 0.05 and _wz_f >= self._sp_f_top + r + 0.005:"""
assert old in src
new = """            # v935: 释放阈值恢复 0.10（滞回）——v930 改 0.05 与触发上限
            # 相同 → 无滞回 swing 反复翻动 → 前轮过伸 1.12 实测。触发
            # <-0.30..0.05，释放 >0.10，中间带稳定
            if _df > 0.10 and _wz_f >= self._sp_f_top + r + 0.005:"""
src = src.replace(old, new, 1)
# 2) swing 腿 kd 环境可调（默认 30，防自由轮过冲）
old2 = """                _kps_d = float(os.environ.get("S10_QP_KP_SW", str(self.kp)))
                if leg in (2, 3):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_REAR",
                                                  str(self.kp)))
                tau[hipy_i] = (_kps_d * (q1t - q1)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kps_d * (q2t - q2)
                               - self.kd * float(qvel[6 + LEG_QV_LEG[b + 2]]))"""
assert old2 in src
new2 = """                _kps_d = float(os.environ.get("S10_QP_KP_SW", str(self.kp)))
                if leg in (2, 3):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_REAR",
                                                  str(self.kp)))
                # v935: swing kd 默认 30（原 8 阻尼比 ~0.25 欠阻尼，轮离地
                # 后自由过冲到 1.1 悬空实测）
                _kds_d = float(os.environ.get("S10_QP_KD_SW", "30.0"))
                tau[hipy_i] = (_kps_d * (q1t - q1)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kps_d * (q2t - q2)
                               - _kds_d * float(qvel[6 + LEG_QV_LEG[b + 2]]))"""
src = src.replace(old2, new2, 1)
p.write_text(src, encoding="utf-8")
print("patched v935")