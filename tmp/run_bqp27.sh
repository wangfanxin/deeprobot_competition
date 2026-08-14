#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_TEST_MAX_SIM=60 S10_STUCK_TIMEOUT=25
bash tmp/run_bench_qp1.sh > tmp/log_bqp27.txt 2>&1
grep -c -e 'QP]' tmp/log_bqp27.txt
grep -e 'VMC-T' -e '侧翻' -e '卡死' -e '完成' -e '到达' tmp/log_bqp27.txt | tail -25
