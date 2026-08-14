#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
grep -n 'KP_POS' tmp/run_real_wbc_ex22.sh
sed -e 's/S10_FP_DRIVE_FLOOR=3.0/S10_FP_DRIVE_FLOOR=3.0 S10_FP_DEBUG=1/' \
    -e 's|tmp/log_real_wbc79.txt|tmp/log_real_wbc80.txt|' \
    tmp/run_real_wbc_ex22.sh > tmp/run_real_wbc_ex23.sh
bash tmp/run_real_wbc_ex23.sh > tmp/log_real_wbc80.txt 2>&1
grep '\[FP\]' tmp/log_real_wbc80.txt | awk 'NR<=10 || NR%50==0' | head -30