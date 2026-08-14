#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
cp tmp/run_real_wbc.sh tmp/run_real_wbc_drop.sh
sed -i "s/export S10_FP_KP_ROLL=400 S10_FP_KP_PITCH=300/export S10_FP_KP_ROLL=400 S10_FP_KP_PITCH=300 S10_FP_STAND_DROP=0.16/" tmp/run_real_wbc_drop.sh
grep STAND_DROP tmp/run_real_wbc_drop.sh
bash tmp/run_real_wbc_drop.sh > tmp/log_real_wbc07.txt 2>&1; echo EXIT=$?