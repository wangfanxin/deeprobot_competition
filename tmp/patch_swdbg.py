#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                _db2 = float(os.environ.get("S10_QP_OV_DB_SW", "0.020"))"""
new = """                if float(os.environ.get("S10_QP_DEBUG", "0")) > 2:
                    print('[SWDBG] t=%.2f leg=%d q1=%.2f q2=%.2f q1t=%.2f '
                          'q2t=%.2f wz=%.3f wzt=%.3f bz=%.3f relx=%.2f '
                          'tauH=%.1f tauK=%.1f'
                          % (self._t, leg, q1, q2, q1t, q2t,
                             wheel_xyz[leg, 2], _wz_t, body["pos"][2],
                             float(_rel[0]), tau[hipy_i], tau[knee_i]),
                          flush=True)
                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t
                _db2 = float(os.environ.get("S10_QP_OV_DB_SW", "0.020"))"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched")