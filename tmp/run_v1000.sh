#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
grep SWING_D tmp/run_real_v998.sh
cp tmp/run_real_v998.sh tmp/run_real_v1000.sh
sed -i "s/S10_STAIR_SWING_D=0.30/S10_STAIR_SWING_D=0.12/" tmp/run_real_v1000.sh
bash tmp/run_real_v1000.sh > tmp/log_real_v1000.txt 2>&1; echo EXIT=$?