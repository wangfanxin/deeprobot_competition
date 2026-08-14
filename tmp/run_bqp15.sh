#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_QP_DEBUG=1 S10_TEST_MAX_SIM=14 S10_STUCK_TIMEOUT=16
bash tmp/run_bench_qp1.sh > tmp/log_bqp15.txt 2>&1
grep -e 'QP]' tmp/log_bqp15.txt | awk '{print }' | sort | uniq -c
