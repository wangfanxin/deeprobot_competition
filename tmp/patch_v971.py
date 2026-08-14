#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# init filters
old = """        self._vx_f = 0.0
        self._om_f = 0.0
        # v961: 抬升目标轮高（外力矩建模用）+ 抬升目标速率（预留）
        self._sw_z_tgt = None
        self._sw_zt = None"""
new = """        self._vx_f = 0.0
        self._om_f = 0.0
        # v961: 抬升目标轮高（外力矩建模用）+ 抬升目标速率（预留）
        self._sw_z_tgt = None
        self._sw_zt = None
        # v971: 防过伸低通滤波（消除 200Hz bang-bang 振荡）
        self._ov_sw_f = np.zeros(4)
        self._ov_st_f = np.zeros(4)"""
assert old in src, "edit0"
src = src.replace(old, new)

# stance anti-over: deadband + low-pass
old = """                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    if _over_s > 0.01:
                        # v957: 支撑腿防过伸用更高增益(默认1000)——前轮悬空
                        # 0.03-0.06 无抓地、狗不前进、RR 进不了窗实测
                        _k_ovs = float(os.environ.get("S10_QP_K_OVER_ST", "1000.0"))
                        tk -= _k_ovs * _over_s
                        if float(os.environ.get("S10_QP_DEBUG", "0")) > 1:
                            print('[OVST] t=%.2f leg=%d wz=%.3f top=%.3f '
                                  'ov=%.3f tk->%.1f'
                                  % (self._t, leg, wheel_xyz[leg, 2], _gt_hi,
                                     _over_s, tk), flush=True)"""
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
                    tk -= self._ov_st_f[leg]
                    if float(os.environ.get("S10_QP_DEBUG", "0")) > 1:
                        print('[OVST] t=%.2f leg=%d wz=%.3f top=%.3f '
                              'ov=%.3f ovf=%.1f tk->%.1f'
                              % (self._t, leg, wheel_xyz[leg, 2], _gt_hi,
                                 _over_s, self._ov_st_f[leg], tk), flush=True)"""
assert old in src, "edit1"
src = src.replace(old, new)

# swing anti-over: deadband + low-pass
old = """                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                if _over2 > 0.02:
                    _k_ov = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                    tau[knee_i] -= _k_ov * _over2
                    if float(os.environ.get("S10_QP_DEBUG", "0")) > 1:
                        print('[OVSW] t=%.2f leg=%d wz=%.3f tgt=%.3f ov=%.3f '
                              'tauK->%.1f'
                              % (self._t, leg, wheel_xyz[leg, 2], _wz_t,
                                 _over2, tau[knee_i]), flush=True)"""
new = """                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                _db2 = float(os.environ.get("S10_QP_OV_DB_SW", "0.020"))
                _ov2_des = 0.0
                if _over2 > _db2:
                    _k_ov = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                    _ov2_des = _k_ov * (_over2 - _db2)
                _lp2 = float(os.environ.get("S10_QP_OV_LP_SW", "0.25"))
                self._ov_sw_f[leg] += _lp2 * (_ov2_des - self._ov_sw_f[leg])
                tau[knee_i] -= self._ov_sw_f[leg]
                if float(os.environ.get("S10_QP_DEBUG", "0")) > 1:
                    print('[OVSW] t=%.2f leg=%d wz=%.3f tgt=%.3f ov=%.3f '
                          'ovf=%.1f tauK->%.1f'
                          % (self._t, leg, wheel_xyz[leg, 2], _wz_t,
                             _over2, self._ov_sw_f[leg], tau[knee_i]),
                          flush=True)"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")