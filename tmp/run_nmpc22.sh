#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_WF=1e-3/S10_NMPC_WF=1e-3 S10_NMPC_KP_YAW=6.0 S10_NMPC_KD_YAW=4.0 S10_NMPC_WM=0.3/' \
    -e 's|tmp/log_nmpc_bench21.txt|tmp/log_nmpc_bench22.txt|' \
    tmp/run_nmpc_bench21.sh > tmp/run_nmpc_bench22.sh
bash tmp/run_nmpc_bench22.sh > tmp/log_nmpc_bench22.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_bench22.txt | tail -10