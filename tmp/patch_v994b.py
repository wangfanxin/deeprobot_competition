#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                if float(wheel_xyz[leg, 2]) > float(terrain_h[leg]) + self.fk.r + 0.03:
                    _rel[0] = min(float(_rel[0]), float(os.environ.get(
                        "S10_QP_TUCK_RELX", "0.06")))"""
new = """                if float(wheel_xyz[leg, 2]) > float(terrain_h[leg]) + self.fk.r + 0.005:
                    _rel[0] = min(float(_rel[0]), float(os.environ.get(
                        "S10_QP_TUCK_RELX", "0.06")))"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched")