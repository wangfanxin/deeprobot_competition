#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# stance anti-over debug
old = """                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    if _over_s > 0.01:
                        # v957: 支撑腿防过伸用更高增益(默认1000)——前轮悬空
                        # 0.03-0.06 无抓地、狗不前进、RR 进不了窗实测
                        _k_ovs = float(os.environ.get("S10_QP_K_OVER_ST", "1000.0"))
                        tk -= _k_ovs * _over_s"""
new = """                if _gt_hi > 0.4:
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
assert old in src, "edit1"
src = src.replace(old, new)

# swing anti-over debug
old = """                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                if _over2 > 0.02:
                    _k_ov = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                    tau[knee_i] -= _k_ov * _over2"""
new = """                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                if _over2 > 0.02:
                    _k_ov = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                    tau[knee_i] -= _k_ov * _over2
                    if float(os.environ.get("S10_QP_DEBUG", "0")) > 1:
                        print('[OVSW] t=%.2f leg=%d wz=%.3f tgt=%.3f ov=%.3f '
                              'tauK->%.1f'
                              % (self._t, leg, wheel_xyz[leg, 2], _wz_t,
                                 _over2, tau[knee_i]), flush=True)"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched")