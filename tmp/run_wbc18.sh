#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed "s/S10_STAIR_SWING_D=0.30/S10_STAIR_SWING_D=0.10/" tmp/run_real_wbc.sh > tmp/run_real_wbc_narrow.sh
bash tmp/run_real_wbc_narrow.sh > tmp/log_real_wbc18.txt 2>&1; echo EXIT=$?