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
                # v940: 支撑腿台面感知位置保持——轮过了 riser 棱(d>0, 在
                # 台面上)时目标封顶到台面顶+r，让前轮在后轮爬顶期压住台面。
                # v936/937 失败是因为宽抬升窗泵高 body 后腿够不到；v939 窄
                # 窗已解决泵高，现在叠加。平地/接近段保持原逻辑。
                _kpp = float(os.environ.get("S10_QP_KP_POS", "80.0"))
                _kdp = float(os.environ.get("S10_QP_KD_POS", "6.0"))
                try:
                    _wzt = float(terrain_h[leg]) + self.fk.r
                    _gt_hi = -1.0
                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        if _dhv <= 0.085:
                            continue
                        _dd = float(np.dot(wheel_xyz[leg, :2] - _rp, _tng))
                        if _dd > 0.0:
                            _gt_hi = max(_gt_hi, float(_top))
                    if _gt_hi > 0.4:
                        _wzt = min(_wzt, _gt_hi + self.fk.r + 0.01)
                    _hip_w2 = body["pos"] + R @ np.array(
                        [0.2277 if leg in (0, 1) else -0.2277, 0.0, 0.0])
                    _relx = (np.cos(yaw) * (wheel_xyz[leg, 0] - _hip_w2[0])
                             + np.sin(yaw) * (wheel_xyz[leg, 1] - _hip_w2[1]))
                    _relz = float(np.clip(_wzt - _hip_w2[2], -0.36, 0.05))
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
print("patched v940")