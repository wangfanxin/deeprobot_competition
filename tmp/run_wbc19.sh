#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_FP_PRESS=0.0
bash tmp/run_real_wbc.sh > tmp/log_real_wbc19.txt 2>&1; echo EXIT=$?