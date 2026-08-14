# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            if stance_mask[leg] > 0.5:
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
assert old in src
new = """            if stance_mask[leg] > 0.5:
                # 接触力 λ（世界）→ body 系 → 矢状面 → Jᵀ
                f_w = lam[leg]
                f_b = R.T @ f_w
                f_s = np.array([float(f_b[0]), -float(f_b[2])])
                th, tk = J.T @ f_s
                # v936: 支撑腿用地形跟随 IK 目标（轮保持 terrain+r）——
                # 原固定 STAND 姿势 PD 在前轮上台面后把轮拉向 0.62（悬空
                # 0.15 实测），2 点支撑无抓地 roll 崩；地形目标让前轮压住
                # 台面顶 0.747，支撑稳
                _kpp = float(os.environ.get("S10_QP_KP_POS", "80.0"))
                _kdp = float(os.environ.get("S10_QP_KD_POS", "6.0"))
                try:
                    _wzt = float(terrain_h[leg]) + self.fk.r
                    _hip_w2 = body["pos"] + R @ np.array(
                        [0.2277 if leg in (0, 1) else -0.2277, 0.0, 0.0])
                    _relx = (np.cos(yaw) * (wheel_xyz[leg, 0] - _hip_w2[0])
                             + np.sin(yaw) * (wheel_xyz[leg, 1] - _hip_w2[1]))
                    _relz = float(np.clip(_wzt - _hip_w2[2], -0.34, 0.05))
                    _q1t, _q2t = q1, q2
                    for _ in range(8):
                        _p = self.fk.wheel_pos(_q1t, _q2t)
                        _err = np.array([_relx - _p[0], _relz + _p[1]])
                        _Jj = self.fk.jac(_q1t, _q2t)
                        _dq = np.linalg.lstsq(_Jj, _err, rcond=None)[0]
                        _dq = np.clip(_dq, -0.2, 0.2)
                        _q1t += float(_dq[0]); _q2t += float(_dq[1])
                        _q1t = float(np.clip(_q1t, -1.7, -0.35))
                        _q2t = float(np.clip(_q2t, -0.2, 3.0))
                    th += (_kpp * (_q1t - q1)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                    tk += (_kpp * (_q2t - q2)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                except Exception:
                    th += (_kpp * (self.pose_target[b + 1] - q1)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                    tk += (_kpp * (self.pose_target[b + 2] - q2)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                tau[hipy_i] = float(np.clip(th, -48, 48))
                tau[knee_i] = float(np.clip(tk, -48, 48))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched qp stance IK v936")