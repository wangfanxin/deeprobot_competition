# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    if _over_s > 0.03:"""
assert old in src
new = """                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    if _over_s > 0.01:"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v956")