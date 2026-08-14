#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_TEST_MAX_SIM=50 S10_STUCK_TIMEOUT=18
bash tmp/run_bench9full.sh > tmp/log_b9o.txt 2>&1
echo EXIT=True
