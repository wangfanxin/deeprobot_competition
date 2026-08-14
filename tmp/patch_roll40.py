# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_QP_AP_K=-20 S10_QP_AR_K=-20 S10_QP_AZ_K=-25 S10_QP_W_R=20 S10_QP_W_P=20", "S10_QP_AP_K=-20 S10_QP_AR_K=-40 S10_QP_AZ_K=-25 S10_QP_W_R=40 S10_QP_W_P=20")
p.write_text(src, encoding="utf-8")
print("patched")