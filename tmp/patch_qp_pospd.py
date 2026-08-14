# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")

# 在 __init__ 加 pose_target
old = "        self.stair_world = []\n        self.stair = None"
assert old in src
new = """        self.stair_world = []
        self.stair = None
        # v910: 支撑腿位置保持目标（STAND 半蹲，与主脚本 STAND_TARGET 一致）
        self.pose_target = np.array([-0.05, -1.10, 1.90,
                                      0.05, -1.10, 1.90,
                                     -0.05,  1.10, -1.90,
                                      0.05,  1.10, -1.90],
                                    dtype=np.float64)"""
src = src.replace(old, new, 1)

# 支撑腿：Jᵀλ + 位置保持 PD
old2 = """            if stance_mask[leg] > 0.5:
                # 接触力 λ（世界）→ body 系 → 矢状面 → Jᵀ
                f_w = lam[leg]
                f_b = R.T @ f_w
                f_s = np.array([float(f_b[0]), -float(f_b[2])])
                th, tk = J.T @ f_s
                tau[hipy_i] = float(np.clip(th, -48, 48))
                tau[knee_i] = float(np.clip(tk, -48, 48))"""
assert old2 in src
new2 = """            if stance_mask[leg] > 0.5:
                # 接触力 λ（世界）→ body 系 → 矢状面 → Jᵀ
                f_w = lam[leg]
                f_b = R.T @ f_w
                f_s = np.array([float(f_b[0]), -float(f_b[2])])
                th, tk = J.T @ f_s
                # v910: 位置保持 PD——纯力控在直腿位形 Jᵀ≈0 无法锁腿构型
                # （台架后腿折叠 body 0.81→0.62 实测）；低增益锁姿势，
                # 力控保持接触力/姿态
                _kpp = float(os.environ.get("S10_QP_KP_POS", "80.0"))
                _kdp = float(os.environ.get("S10_QP_KD_POS", "6.0"))
                th += (_kpp * (self.pose_target[b + 1] - q1)
                       - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tk += (_kpp * (self.pose_target[b + 2] - q2)
                       - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                tau[hipy_i] = float(np.clip(th, -48, 48))
                tau[knee_i] = float(np.clip(tk, -48, 48))"""
src = src.replace(old2, new2, 1)

p.write_text(src, encoding="utf-8")
print("patched")