#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_KP_Z=200/S10_NMPC_KP_Z=100/' \
    -e 's|tmp/log_nmpc_bench9.txt|tmp/log_nmpc_bench10.txt|' \
    tmp/run_nmpc_bench7.sh > tmp/run_nmpc_bench10.sh
bash tmp/run_nmpc_bench10.sh > tmp/log_nmpc_bench10.txt 2>&1
grep 'NMPC\]\|VMC-T\|侧翻\|FREQ\|卡死' tmp/log_nmpc_bench10.txt | head -18