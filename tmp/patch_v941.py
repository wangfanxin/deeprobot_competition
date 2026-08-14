# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                _kpp = float(os.environ.get("S10_QP_KP_POS", "80.0"))
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
assert old in src
new = """                _kpp = float(os.environ.get("S10_QP_KP_POS", "80.0"))
                _kdp = float(os.environ.get("S10_QP_KD_POS", "6.0"))
                # v941: 支撑腿 IK 只在轮真正过了 riser 棱(d>0, 在台面上)
                # 时启用——v940 全时段 IK 在接近段与 λ 冲突振荡(多解)翻车
                # 实测；接近段保持姿势 PD(稳定)。
                _gt_hi = -1.0
                try:
                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        if _dhv <= 0.085:
                            continue
                        _dd = float(np.dot(wheel_xyz[leg, :2] - _rp, _tng))
                        if _dd > 0.0:
                            _gt_hi = max(_gt_hi, float(_top))
                except Exception:
                    pass
                if _gt_hi > 0.4:
                    try:
                        _wzt = min(float(terrain_h[leg]) + self.fk.r,
                                   _gt_hi + self.fk.r + 0.01)
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
                else:
                    th += (_kpp * (self.pose_target[b + 1] - q1)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                    tk += (_kpp * (self.pose_target[b + 2] - q2)
                           - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                tau[hipy_i] = float(np.clip(th, -48, 48))
                tau[knee_i] = float(np.clip(tk, -48, 48))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v941")