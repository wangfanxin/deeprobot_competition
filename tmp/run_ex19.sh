#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_STAND_DROP=0.22/S10_FP_STAND_DROP=0.22 S10_FP_DEBUG=1/' \
    -e 's|tmp/log_real_wbc75.txt|tmp/log_real_wbc76.txt|' \
    tmp/run_real_wbc_ex18.sh > tmp/run_real_wbc_ex19.sh
bash tmp/run_real_wbc_ex19.sh > tmp/log_real_wbc76.txt 2>&1
grep '\[FP\]' tmp/log_real_wbc76.txt | awk 'NR % 20 == 0' | head -40