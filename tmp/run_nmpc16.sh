#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_KP_SW=120/S10_NMPC_KP_SW=120 S10_NMPC_KD_SW=15/' \
    -e 's|tmp/log_nmpc_bench15.txt|tmp/log_nmpc_bench16.txt|' \
    tmp/run_nmpc_bench15.sh > tmp/run_nmpc_bench16.sh
bash tmp/run_nmpc_bench16.sh > tmp/log_nmpc_bench16.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_bench16.txt | tail -16