#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
cp tmp/run_real_qp5.sh tmp/run_real_qp6.sh
sed -i "s/export S10_STAIR_BENCH=0/export S10_STAIR_BENCH=0 S10_STAIR_RISER_MIN=0.15/" tmp/run_real_qp6.sh
grep RISER tmp/run_real_qp6.sh
bash tmp/run_real_qp6.sh > tmp/log_real05.txt 2>&1; echo EXIT=$?