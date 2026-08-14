#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_FP_KPH=600
bash tmp/run_real_wbc_r18.sh > tmp/log_real_wbc46.txt 2>&1; echo EXIT=$?