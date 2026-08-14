#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_QP_DEBUG=1 S10_TEST_MAX_SIM=8 S10_STUCK_TIMEOUT=12
bash tmp/run_bench_qp1.sh > tmp/log_bqp22.txt 2>&1
grep -e 'QP]' tmp/log_bqp22.txt | sed 's/.*st=/st=/' | sed 's/ ad=.*//' | sort | uniq -c
