#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_TEST_MAX_SIM=35 S10_STUCK_TIMEOUT=15
bash tmp/run_bench13.sh > tmp/log_b13b.txt 2>&1
echo EXIT=True
