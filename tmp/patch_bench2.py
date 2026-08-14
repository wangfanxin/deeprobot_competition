# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_STAIR_SWING_D=0.15", "S10_STAIR_SWING_D=0.25")
src = src.replace("export S10_FP_KP_POS=120", "export S10_FP_KP_POS=120\nexport S10_QP_KP_SW=100 S10_QP_KP_HIPX=100 S10_QP_AP_K=-8 S10_QP_AR_K=-15")
p.write_text(src, encoding="utf-8")
print("patched")