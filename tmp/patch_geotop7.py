# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
old = """                            if _di <= 0.0 and float(_top) > _geo_top4[_i]:
                                _geo_top4[_i] = float(_top)"""
assert old in src
new = """                            # v927: 符号修正——d>0 才是"过了该棱、在台面上"
                            # （原 d<=0 把棱前轮也算在台面上 → 后轮在棱前被
                            # 拉到 0.747 折叠实测）
                            if _di > 0.0 and float(_top) > _geo_top4[_i]:
                                _geo_top4[_i] = float(_top)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")