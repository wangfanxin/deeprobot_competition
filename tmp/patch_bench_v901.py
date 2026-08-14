# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench9full.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_STAIR_WIN_VX=0.8 S10_STAIR_LIFT_HI=0.10", "S10_STAIR_WIN_VX=1.8")
src = src.replace("S10_FP_KP_POS=150", "S10_FP_KP_POS=200")
p.write_text(src, encoding="utf-8")
print(src)