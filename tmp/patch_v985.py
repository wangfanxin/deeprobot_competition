#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# ---- swing anti-over: Jacobian-projected ----
old = """                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                _db2 = float(os.environ.get("S10_QP_OV_DB_SW", "0.020"))
                _ov2_des = 0.0
                if _over2 > _db2:
                    _k_ov = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                    _ov2_des = _k_ov * (_over2 - _db2)
                _lp2 = float(os.environ.get("S10_QP_OV_LP_SW", "0.25"))
                self._ov_sw_f[leg] += _lp2 * (_ov2_des - self._ov_sw_f[leg])
                tau[knee_i] -= self._ov_sw_f[leg]"""
new = """                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                _db2 = float(os.environ.get("S10_QP_OV_DB_SW", "0.020"))
                _ov2_des = 0.0
                if _over2 > _db2:
                    _k_ov = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                    _ov2_des = _k_ov * (_over2 - _db2)
                _lp2 = float(os.environ.get("S10_QP_OV_LP_SW", "0.25"))
                self._ov_sw_f[leg] += _lp2 * (_ov2_des - self._ov_sw_f[leg])
                # v985: 防过伸力矩经 Jacobian 投影到 hipy+knee——轮折叠到
                # q1+q2≈π 时膝盖对轮高近奇异(无权威)，只打膝盖推不动轮
                # (悬空 0.9+ 实测)。J^T [0, -F] 让大腿也参与压轮。
                if abs(self._ov_sw_f[leg]) > 0.5:
                    _tov = J.T @ np.array([0.0, -self._ov_sw_f[leg]])
                    tau[hipy_i] += float(_tov[0])
                    tau[knee_i] += float(_tov[1])"""
assert old in src, "edit1"
src = src.replace(old, new)

# ---- stance anti-over: Jacobian-projected ----
old = """                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    # v971: 死区(0.01)+低通滤波——原 K=1000 全幅反馈在
                    # 200Hz 下 bang-bang 振荡(±48 高频抖动，轮悬空乱颤)
                    _db = float(os.environ.get("S10_QP_OV_DB", "0.010"))
                    _ov_des = 0.0
                    if _over_s > _db:
                        _ov_des = float(os.environ.get(
                            "S10_QP_K_OVER_ST", "1000.0")) * (_over_s - _db)
                    _lp = float(os.environ.get("S10_QP_OV_LP", "0.25"))
                    self._ov_st_f[leg] += _lp * (_ov_des - self._ov_st_f[leg])
                    tk -= self._ov_st_f[leg]"""
new = """                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    # v971: 死区(0.01)+低通滤波——原 K=1000 全幅反馈在
                    # 200Hz 下 bang-bang 振荡(±48 高频抖动，轮悬空乱颤)
                    _db = float(os.environ.get("S10_QP_OV_DB", "0.010"))
                    _ov_des = 0.0
                    if _over_s > _db:
                        _ov_des = float(os.environ.get(
                            "S10_QP_K_OVER_ST", "1000.0")) * (_over_s - _db)
                    _lp = float(os.environ.get("S10_QP_OV_LP", "0.25"))
                    self._ov_st_f[leg] += _lp * (_ov_des - self._ov_st_f[leg])
                    # v985: 经 Jacobian 投影(同 swing)——折叠位姿下大腿参与
                    th += float((J.T @ np.array([0.0, -self._ov_st_f[leg]]))[0])
                    tk += float((J.T @ np.array([0.0, -self._ov_st_f[leg]]))[1])"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")