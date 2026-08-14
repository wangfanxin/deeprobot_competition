#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_FP_STAND_DROP=0.16
bash tmp/run_real_wbc.sh > tmp/log_real_wbc08.txt 2>&1; echo EXIT=$?