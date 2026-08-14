#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_WM=0.05/S10_NMPC_WM=0.3 S10_NMPC_PITCH_FF_SW=1.5/' \
    -e 's|tmp/log_nmpc_bench37.txt|tmp/log_nmpc_bench39.txt|' \
    tmp/run_nmpc_bench37.sh > tmp/run_nmpc_bench39.sh
grep -n 'WM=\|PITCH_FF' tmp/run_nmpc_bench39.sh
bash tmp/run_nmpc_bench39.sh > tmp/log_nmpc_bench39.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_bench39.txt | tail -12