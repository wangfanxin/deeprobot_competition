#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_KP_SW=220/S10_NMPC_KP_SW=120/' \
    -e 's|tmp/log_nmpc_bench14.txt|tmp/log_nmpc_bench15.txt|' \
    tmp/run_nmpc_bench10.sh > tmp/run_nmpc_bench15.sh
bash tmp/run_nmpc_bench15.sh > tmp/log_nmpc_bench15.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_bench15.txt | tail -18