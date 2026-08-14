#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_BODY_KD=0.2/S10_FP_BODY_KD=0.2 S10_FP_DEBUG=1/' \
    -e 's|tmp/log_real_wbc81.txt|tmp/log_real_wbc82.txt|' \
    tmp/run_real_wbc_ex24.sh > tmp/run_real_wbc_ex25.sh
bash tmp/run_real_wbc_ex25.sh > tmp/log_real_wbc82.txt 2>&1
grep '\[FP\]' tmp/log_real_wbc82.txt | awk 'NR<=16' 