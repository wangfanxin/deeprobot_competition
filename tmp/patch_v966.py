#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """            else:
                if (d[i] > 0.08 and wz[i] >= self._sp_top[i] + r + 0.005
                        and not self._done[i]):"""
new = """            else:
                # v966: 释放放宽到轮心过棱(d>0.02)——原 d>0.08 让轮在台面
                # 上方悬空 2-3cm 等 d 推进(无抓地、狗不前进 3-4s 实测)；
                # 释放后支撑分支会压台面 2mm 恢复抓地
                if (d[i] > 0.02 and wz[i] >= self._sp_top[i] + r - 0.005
                        and not self._done[i]):"""
assert old in src, "anchor missing"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")