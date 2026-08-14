#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_WA=0.0/S10_NMPC_WA=1.0/' \
    -e 's/S10_NMPC_WM=0.0/S10_NMPC_WM=0.05/' \
    -e 's|tmp/log_nmpc_bench20.txt|tmp/log_nmpc_bench21.txt|' \
    tmp/run_nmpc_bench20.sh > tmp/run_nmpc_bench21.sh
bash tmp/run_nmpc_bench21.sh > tmp/log_nmpc_bench21.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_bench21.txt | tail -12