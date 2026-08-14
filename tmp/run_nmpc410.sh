#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_WM=0.3/S10_NMPC_WM=0.3 S10_NMPC_KP_Z=300 S10_NMPC_DEBUG=1/' \
    -e 's|tmp/log_nmpc_real8.txt|tmp/log_nmpc_real10.txt|' \
    tmp/run_nmpc_real8.sh > tmp/run_nmpc_real10.sh
bash tmp/run_nmpc_real10.sh > tmp/log_nmpc_real10.txt 2>&1
grep 'NMPC\]' tmp/log_nmpc_real10.txt | tail -6
grep 'VMC-T\|侧翻' tmp/log_nmpc_real10.txt | tail -4