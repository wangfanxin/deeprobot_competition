# -*- coding: utf-8 -*-
import pathlib, re
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench13.sh")
src = p.read_text(encoding="utf-8")
if "S10_VMC_TERRAIN_KIN" not in src:
    src = re.sub(r"(export S10_YAW_DAMP=.*\n)", r"\1export S10_VMC_TERRAIN_KIN=1\n", src, count=1)
p.write_text(src, encoding="utf-8")
print("patched:", "S10_VMC_TERRAIN_KIN" in src)