# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
old = "export S10_QP_KP_SW=100 S10_QP_KP_HIPX=100 S10_QP_AP_K=0 S10_QP_AR_K=0 S10_QP_AZ_K=-25"
assert old in src, "qp line"
new = "export S10_QP_KP_SW=100 S10_QP_KP_HIPX=100 S10_QP_AP_K=-20 S10_QP_AR_K=-20 S10_QP_AZ_K=-25 S10_QP_W_R=20 S10_QP_W_P=20"
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")