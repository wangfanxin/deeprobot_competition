# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            if _df > 0.10 and _wz_f >= self._sp_f_top + r + 0.005:"""
assert old in src
new = """            # v930: 前轮释放阈值 0.10→0.05——d∈(0.05,0.10) 带里前轮反复
            # 重触发阻塞后轮 SWING（QP 台架 y38.6 前轮上台面但后轮不爬
            # 实测）；放宽后前轮早释放、后轮可触发
            if _df > 0.05 and _wz_f >= self._sp_f_top + r + 0.005:"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched qp release")