# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench9full.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("export S10_FP_YAW_KP=2.0 S10_FP_YAW_KD=0.5", "export S10_FP_YAW_KP=2.0 S10_FP_YAW_KD=0.5\nexport S10_FP_KP_HIPX=8 S10_FP_KD_HIPX=2", 1)
p.write_text(src, encoding="utf-8")
print("patched")