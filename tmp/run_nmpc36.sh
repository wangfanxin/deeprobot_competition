#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_WM=0.0/S10_NMPC_WM=0.1/' \
    -e 's/S10_NMPC_KP_YAW=6.0/S10_NMPC_KP_YAW=10.0/' \
    -e 's|tmp/log_nmpc_bench35.txt|tmp/log_nmpc_bench36.txt|' \
    tmp/run_nmpc_bench35.sh > tmp/run_nmpc_bench36.sh
grep -n 'KP_YAW\|WM=' tmp/run_nmpc_bench36.sh
bash tmp/run_nmpc_bench36.sh > tmp/log_nmpc_bench36.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_bench36.txt | tail -10