# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_QP_K_OVER=300", "S10_QP_K_OVER=500")
p.write_text(src, encoding="utf-8")
print("patched")