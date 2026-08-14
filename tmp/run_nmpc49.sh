#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_KP_Z=100/S10_NMPC_KP_Z=300/' \
    -e 's|tmp/log_nmpc_real8.txt|tmp/log_nmpc_real9.txt|' \
    tmp/run_nmpc_real8.sh > tmp/run_nmpc_real9.sh
grep -n 'KP_Z' tmp/run_nmpc_real9.sh
bash tmp/run_nmpc_real9.sh > tmp/log_nmpc_real9.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_real9.txt | tail -12