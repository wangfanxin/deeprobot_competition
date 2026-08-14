#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# remove the broken [STIK] print (references undefined _q1t/_q2t before IK)
old = """                        _wzt = min(float(terrain_h[leg]) + self.fk.r,
                                   _gt_hi + self.fk.r - 0.002)
                        if float(os.environ.get("S10_QP_DEBUG", "0")) > 1:
                            print('[STIK] t=%.2f leg=%d bz=%.3f wz=%.3f '
                                  'wzt=%.3f q1=%.2f q2=%.2f q1t=%.2f q2t=%.2f'
                                  % (self._t, leg, body["pos"][2],
                                     wheel_xyz[leg, 2], _wzt, q1, q2,
                                     _q1t, _q2t), flush=True)"""
new = """                        _wzt = min(float(terrain_h[leg]) + self.fk.r,
                                   _gt_hi + self.fk.r - 0.002)"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")