#!/usr/bin/env python3
import io

# add joint debug in FootPlaceVMC stance branch (after q1t/q2t computed)
p1 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py"
s1 = io.open(p1, "r", encoding="utf-8").read()
old = """            q1t, q2t = self._ik(float(rel[0]), _rz, q1, q2,
                                 lift=(sl > 0.1), leg=leg)"""
new = """            q1t, q2t = self._ik(float(rel[0]), _rz, q1, q2,
                                 lift=(sl > 0.1), leg=leg)
            if os.environ.get("S10_FP_DEBUG", "0") == "1" and leg == 0:
                print("[LEGDBG] t=%.2f leg=%d bz=%.3f bdes=%.3f wz=%.3f "
                      "hipz=%.3f relz=%.3f q1=%.2f q2=%.2f q1t=%.2f q2t=%.2f"
                      % (self._t, leg, body["pos"][2], _bdes_z, wz,
                         hip_w[2], _rz, q1, q2, q1t, q2t), flush=True)"""
assert old in s1, "anchor"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)
print("patched")