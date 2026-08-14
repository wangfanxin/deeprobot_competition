#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_FP_BODY_K=0.8
export S10_FP_STAND_DROP=0.18
bash tmp/run_real_wbc.sh > tmp/log_real_wbc16.txt 2>&1; echo EXIT=$?