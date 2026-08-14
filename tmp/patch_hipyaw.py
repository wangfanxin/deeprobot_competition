# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_FP_YAW_KP=4.0 S10_FP_YAW_KD=0.8", "S10_FP_YAW_KP=10.0 S10_FP_YAW_KD=2.5")
p.write_text(src, encoding="utf-8")
print("patched")