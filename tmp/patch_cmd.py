# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
old = """                      climb_mask=_climb_mask,
                      place_z=place_z,
                      place_margin=float(os.environ.get(
                          'S10_STAIR_LIFT_MARGIN', '0.04')))"""
assert old in src
new = """                      climb_mask=_climb_mask,
                      place_z=place_z,
                      geo_top=_geo_top4,
                      place_margin=float(os.environ.get(
                          'S10_STAIR_LIFT_MARGIN', '0.04')))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched cmd")