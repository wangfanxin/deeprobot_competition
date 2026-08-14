# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_real_qp.sh")
src = p.read_text(encoding="utf-8")
old = "export S10_QP_KP_SW=100 S10_QP_KP_SW_REAR=200 S10_QP_KP_POS=150 S10_QP_KP_HIPX=100\nexport S10_QP_AP_K=-20 S10_QP_AR_K=-20 S10_QP_AZ_K=-25 S10_QP_W_R=20 S10_QP_W_P=20"
assert old in src
new = "export S10_QP_KP_SW=100 S10_QP_KD_SW=30 S10_QP_KP_SW_REAR=80 S10_QP_KP_POS=150 S10_QP_KP_HIPX=100\nexport S10_QP_AP_K=-20 S10_QP_AR_K=-40 S10_QP_AZ_K=-25 S10_QP_W_R=40 S10_QP_W_P=20\nexport S10_QP_K_OVER=500 S10_QP_K_OVER_ST=1000"
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched real qp")