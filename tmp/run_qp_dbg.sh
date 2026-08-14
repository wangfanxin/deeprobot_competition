#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_QP_DEBUG=1
export S10_TEST_MAX_SIM=6.5 S10_STUCK_TIMEOUT=15
bash tmp/run_bench_qp1.sh > tmp/log_qp_dbg.txt 2>&1
echo EXIT=True
