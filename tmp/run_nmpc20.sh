#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_WA=1.0/S10_NMPC_WA=0.0/' \
    -e 's/S10_NMPC_WF=1e-3/S10_NMPC_WF=1e-3 S10_NMPC_WM=0.0/' \
    -e 's|tmp/log_nmpc_bench19.txt|tmp/log_nmpc_bench20.txt|' \
    tmp/run_nmpc_bench19.sh > tmp/run_nmpc_bench20.sh
bash tmp/run_nmpc_bench20.sh > tmp/log_nmpc_bench20.txt 2>&1
grep 'VMC-T\|侧翻\|卡死' tmp/log_nmpc_bench20.txt | tail -10