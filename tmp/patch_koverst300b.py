# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
old = "S10_QP_K_OVER=500 S10_QP_KP_POS=150"
assert old in src
new = "S10_QP_K_OVER=500 S10_QP_K_OVER_ST=300 S10_QP_KP_POS=150"
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")