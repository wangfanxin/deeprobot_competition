# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench13.sh")
src = p.read_text(encoding="utf-8")
old = "S10_STAIR_WIN_VX=1.8"
assert old in src
src = src.replace(old, "S10_STAIR_WIN_VX=1.0", 1)
p.write_text(src, encoding="utf-8")
print("patched")