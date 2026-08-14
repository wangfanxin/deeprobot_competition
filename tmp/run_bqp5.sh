#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_TEST_MAX_SIM=60 S10_STUCK_TIMEOUT=25
bash tmp/run_bench_qp1.sh > tmp/log_bqp5.txt 2>&1
echo EXIT=True
