#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# Add closed-form IK method after _face_place_z (before QP section)
old = """    # ---------------- QP：接触力分配（SRBD） ----------------
    def _qp_solve(self, body, wheel_xyz, stance_mask, lam_ref):"""
new = """    # ---------------- 闭式 2 连杆 IK（v979） ----------------
    def _ik_closed(self, xd, zd):
        \"\"\"给定 body 系轮心目标 (xd=前向, zd=向下为正)，返回 (q1, q2)。

        替代 8 次牛顿迭代——迭代从当前(折叠)姿态出发会收敛到"向上折叠"
        错误分支(q2≈2.8 轮越过髋、悬空 0.9+ 实测)。闭式解取 q2=+acos
        (自然膝弯曲分支)，一次到位。
        \"\"\"
        L1, L2 = self.fk.L1, self.fk.L2
        r2 = xd * xd + zd * zd
        r2 = min(r2, (L1 + L2) ** 2 - 1e-6)
        c2 = float(np.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2),
                           -1.0, 1.0))
        q2 = float(np.arccos(c2))
        q1 = float(np.arctan2(xd, zd) - np.arctan2(
            L2 * np.sin(q2), L1 + L2 * np.cos(q2)))
        return q1, q2

    # ---------------- QP：接触力分配（SRBD） ----------------
    def _qp_solve(self, body, wheel_xyz, stance_mask, lam_ref):"""
assert old in src, "edit0"
src = src.replace(old, new)

# Replace stance branch Newton IK with closed-form
old = """                        _relz = float(np.clip(_wzt - _hip_w2[2], -0.36, 0.05))
                        _q1t, _q2t = q1, q2
                        for _ in range(8):
                            _p = self.fk.wheel_pos(_q1t, _q2t)
                            _err = np.array([_relx - _p[0], _relz + _p[1]])
                            _Jj = self.fk.jac(_q1t, _q2t)
                            _dq = np.linalg.lstsq(_Jj, _err, rcond=None)[0]
                            _dq = np.clip(_dq, -0.2, 0.2)
                            _q1t += float(_dq[0]); _q2t += float(_dq[1])
                            # v974/v975: q1 上界放宽到 1.0——台面支撑解需要
                            # 大腿更水平(q1≈0.7)，0.3 仍顶到边界；物理 ±2.53
                            _q1t = float(np.clip(_q1t, -1.7, 1.0))
                            # v978: q2 下界放宽到 -1.0——0.5 让短垂距解(轮在
                            # 台面、body 0.83 只需 8cm 垂距)不可达，IK 被迫
                            # 向上折叠(q2≈2.8 轮越过髋到 0.91 悬空实测)
                            _q2t = float(np.clip(_q2t, -1.0, 3.0))"""
new = """                        _relz = float(np.clip(_wzt - _hip_w2[2], -0.36, 0.05))
                        # v979: 闭式 IK——迭代式从折叠姿态出发收敛到错误分支
                        _zd_t = max(0.001, -_relz)
                        _q1t, _q2t = self._ik_closed(_relx, _zd_t)
                        _q1t = float(np.clip(_q1t, -1.7, 1.0))
                        _q2t = float(np.clip(_q2t, -1.0, 3.0))"""
assert old in src, "edit1"
src = src.replace(old, new)

# Replace swing branch Newton IK with closed-form
old = """                _rz = float(np.clip(_rel[1], -0.34, 0.15))
                q1t, q2t = q1, q2
                for _ in range(8):
                    p = self.fk.wheel_pos(q1t, q2t)
                    err = np.array([_rel[0] - p[0], _rz + p[1]])
                    Jj = self.fk.jac(q1t, q2t)
                    dq = np.linalg.lstsq(Jj, err, rcond=None)[0]
                    dq = np.clip(dq, -0.25, 0.25)
                    q1t += float(dq[0]); q2t += float(dq[1])
                    # v929: SWING 腿 q1 正常分支（防镜像折叠过伸，同 FP v925）
                    # v974: q1 上界放宽到 0.2(台面贴面解需要)，下界 -1.1 保留
                    q1t = float(np.clip(q1t, -1.1, 0.5))
                    # v978: q2 下界 -1.0（同支撑腿）——0.5 强制向上折叠
                    q2t = float(np.clip(q2t, -1.0, 3.0))"""
new = """                _rz = float(np.clip(_rel[1], -0.34, 0.15))
                # v979: 闭式 IK(同支撑腿)——迭代式易收敛到向上折叠分支
                _zd_s = max(0.001, -_rz)
                q1t, q2t = self._ik_closed(float(_rel[0]), _zd_s)
                # v929: 防镜像折叠过伸(下界 -1.1 保留)；上界放宽
                q1t = float(np.clip(q1t, -1.1, 0.5))
                q2t = float(np.clip(q2t, -1.0, 3.0))"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")