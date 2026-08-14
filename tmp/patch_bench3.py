# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_QP_AP_K=-8 S10_QP_AR_K=-15", "S10_QP_AP_K=0 S10_QP_AR_K=0 S10_QP_AZ_K=-25")
p.write_text(src, encoding="utf-8")
print("patched")