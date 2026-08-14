#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_SWING_D=0.35/S10_NMPC_SWING_D=0.01/' \
    -e 's|tmp/log_nmpc_bench18.txt|tmp/log_nmpc_bench19.txt|' \
    tmp/run_nmpc_bench16.sh > tmp/run_nmpc_bench19.sh
bash tmp/run_nmpc_bench19.sh > tmp/log_nmpc_bench19.txt 2>&1
grep 'VMC-T\|侧翻\|卡死' tmp/log_nmpc_bench19.txt | tail -12