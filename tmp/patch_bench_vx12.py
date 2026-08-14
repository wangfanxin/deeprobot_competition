# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench9full.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_STAIR_WIN_VX=1.8", "S10_STAIR_WIN_VX=1.2", 1)
p.write_text(src, encoding="utf-8")
print("patched")