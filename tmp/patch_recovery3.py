#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """            if _recov_on:
                for _leg in (2, 3):
                    tau[WHEEL_Q_IDX[_leg]] = -13.5
                for _leg in (0, 1):
                    if float(step_lift[_leg]) <= 0.5:
                        tau[WHEEL_Q_IDX[_leg]] = -3.0"""
new = """            if _recov_on:
                _rdrive = -float(os.environ.get("S10_FP_RECOV_DRIVE", "8.0"))
                for _leg in (2, 3):
                    tau[WHEEL_Q_IDX[_leg]] = _rdrive
                for _leg in (0, 1):
                    if float(step_lift[_leg]) <= 0.5:
                        tau[WHEEL_Q_IDX[_leg]] = -3.0"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")