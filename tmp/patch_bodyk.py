# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench9full.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_FP_BODY_K=0.0", "S10_FP_BODY_K=0.4")
p.write_text(src, encoding="utf-8")
print("patched")