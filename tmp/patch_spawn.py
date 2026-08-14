# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")

# 1) bench 模式默认用台架 XML
old = "XML = os.environ.get('S10_XML',\n    f'{PKG}/S10_description/s10_mjcf/mjcf/S10_track.xml')"
assert old in src
new = """XML = os.environ.get('S10_XML', '')
if not XML:
    _bf = os.environ.get('S10_STAIR_BENCH', '0')
    XML = (f'{PKG}/S10_description/s10_mjcf/mjcf/S10_track_bench.xml'
           if float(_bf) > 0
           else f'{PKG}/S10_description/s10_mjcf/mjcf/S10_track.xml')"""
src = src.replace(old, new, 1)

# 2) 台架出生点：平台上 y=37.34（riser2 前 1.0m），z=0.78（0.54 平台 + 0.24）
old2 = "            d.qpos[0:3] = [-14.50, 37.14, 0.72]"
assert old2 in src
new2 = "            d.qpos[0:3] = [-14.50, 37.34, 0.78]"
src = src.replace(old2, new2, 1)

p.write_text(src, encoding="utf-8")
print("patched")