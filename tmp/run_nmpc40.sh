#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_WM=0.3/S10_NMPC_WM=0.5 S10_NMPC_PITCH_FF_SW=3.0/' \
    -e 's|tmp/log_nmpc_bench39.txt|tmp/log_nmpc_bench40.txt|' \
    tmp/run_nmpc_bench39.sh > tmp/run_nmpc_bench40.sh
bash tmp/run_nmpc_bench40.sh > tmp/log_nmpc_bench40.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_bench40.txt | tail -12