#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
cp tmp/run_real_wbc.sh tmp/run_real_wbc_k0.sh
sed -i "s/S10_FP_WHEEL_K=4 S10_FP_WHEEL_D=0.08 S10_FP_KP_POS=220 S10_FP_KP_ROLL=400 S10_FP_KP_PITCH=300/S10_FP_WHEEL_K=4 S10_FP_WHEEL_D=0.08 S10_FP_KP_POS=220 S10_FP_KP_ROLL=400 S10_FP_KP_PITCH=300 S10_FP_BODY_K=0.0/" tmp/run_real_wbc_k0.sh
grep BODY_K tmp/run_real_wbc_k0.sh
bash tmp/run_real_wbc_k0.sh > tmp/log_real_wbc06.txt 2>&1; echo EXIT=$?