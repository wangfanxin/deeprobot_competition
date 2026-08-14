# -*- coding: utf-8 -*-
import pathlib, re
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
qp_line = "export S10_QP_KP_SW=100 S10_QP_KP_HIPX=100 S10_QP_AP_K=0 S10_QP_AR_K=0 S10_QP_AZ_K=-25"
if "S10_QP_KP_SW" not in src:
    # insert after the S10_FP_WHEEL_K line
    src = re.sub(r"(export S10_FP_WHEEL_K=.*\n)", r"\1" + qp_line + "\n", src, count=1)
p.write_text(src, encoding="utf-8")
print("patched:", "S10_QP_KP_SW" in src)