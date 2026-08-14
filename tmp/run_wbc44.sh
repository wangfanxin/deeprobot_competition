#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_FP_STAND_DROP=0.20
bash tmp/run_real_wbc_r18.sh > tmp/log_real_wbc44.txt 2>&1; echo EXIT=$?