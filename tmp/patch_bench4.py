# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
anchor = "export S10_FP_KP_POS=120 S10_FP_KD=8"
assert anchor in src
qp_line = "export S10_QP_KP_SW=100 S10_QP_KP_HIPX=100 S10_QP_AP_K=0 S10_QP_AR_K=0 S10_QP_AZ_K=-25"
if "S10_QP_KP_SW" not in src:
    src = src.replace(anchor, anchor + "\n" + qp_line, 1)
p.write_text(src, encoding="utf-8")
print("patched, qp lines:", "S10_QP_KP_SW" in src)