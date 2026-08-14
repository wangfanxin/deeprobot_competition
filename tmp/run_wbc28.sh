#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
grep -n "swing_d = " src/S10_sdk_deploy/s10_mpc/stair_wbc.py
export S10_FP_BODY_PD=0.15
export S10_FP_KP_PITCH=500
bash tmp/run_real_wbc.sh > tmp/log_real_wbc28.txt 2>&1; echo EXIT=$?