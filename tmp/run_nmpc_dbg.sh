#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_NMPC_WF=1e-3/S10_NMPC_WF=1e-3 S10_NMPC_DEBUG=1/' \
    -e 's|tmp/log_nmpc_bench6.txt|tmp/log_nmpc_bench7.txt|' \
    tmp/run_nmpc_bench.sh > tmp/run_nmpc_bench7.sh
bash tmp/run_nmpc_bench7.sh > tmp/log_nmpc_bench7.txt 2>&1
grep 'NMPC\]\|VMC-T\|侧翻' tmp/log_nmpc_bench7.txt | head -20