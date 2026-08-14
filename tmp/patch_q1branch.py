# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py")
src = p.read_text(encoding="utf-8-sig")
old = """        _q1_lo, _q1_hi = (-0.35, 0.9) if lift else (-1.7, -0.35)"""
assert old in src
new = """        # v925: SWING 腿 q1 用正常分支 [-1.1,-0.3]——原 [-0.35,0.9] 允许
        # 镜像折叠解（q1=0.9 轮在髋上方 0.06m，贴面爬升时过伸 1.0+ 悬空
        # 实测）。贴面爬升轮应保持在髋下方（正常分支），q2 放宽到 0.5 已
        # 足够伸到台面顶。
        _q1_lo, _q1_hi = (-1.1, -0.3) if lift else (-1.7, -0.35)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")