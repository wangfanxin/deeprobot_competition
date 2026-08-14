# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench13.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_STAIR_VX_RAMP=4.0", "S10_STAIR_VX_RAMP=4.0 S10_STAIR_SWING_WHEEL0=0", 1)
p.write_text(src, encoding="utf-8")
print("patched")