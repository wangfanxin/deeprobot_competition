# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                    try:
                        _wzt = min(float(terrain_h[leg]) + self.fk.r,
                                   _gt_hi + self.fk.r + 0.01)"""
assert old in src
new = """                    try:
                        # v959: 支撑腿目标压入台面 2mm（原 +0.01 余量让轮
                        # 悬空 0.01-0.06 无抓地、狗不推进实测）
                        _wzt = min(float(terrain_h[leg]) + self.fk.r,
                                   _gt_hi + self.fk.r - 0.002)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v959")