#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                if _ok:
                    self._sp[i] = 1.0
                    self._sp_top[i] = top[i]
                    self._rel_t[i] = None"""
new = """                if _ok:
                    if float(os.environ.get("S10_QP_DEBUG", "0")) > 0:
                        print('[PHASE] t=%.2f trig leg=%d d=%.3f top=%.3f '
                              'wz=%.3f swd=%.2f done=%s'
                              % (self._t, i, d[i], top[i], wz[i], _swd,
                                 str(self._done)), flush=True)
                    self._sp[i] = 1.0
                    self._sp_top[i] = top[i]
                    self._rel_t[i] = None"""
assert old in src, "anchor missing"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched")