#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_QP_DEBUG=0 S10_TEST_MAX_SIM=35 S10_STUCK_TIMEOUT=15
bash tmp/run_bench_qp1.sh > tmp/log_qp8.txt 2>&1
echo EXIT=True
