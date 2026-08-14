#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed "s/S10_STAIR_VX_RAMP=10.0/S10_STAIR_VX_RAMP=15.0/" tmp/run_real_wbc.sh > tmp/run_real_wbc_ramp15.sh
bash tmp/run_real_wbc_ramp15.sh > tmp/log_real_wbc32.txt 2>&1; echo EXIT=$?