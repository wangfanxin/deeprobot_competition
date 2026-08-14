#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_YAW_DIFF=4.0/S10_FP_YAW_DIFF=4.0 S10_NAV_DEBUG=1/' \
    -e 's|tmp/log_real_wbc91.txt|tmp/log_real_wbc92.txt|' \
    -e 's|tmp/traj_wbc91.npy|tmp/traj_wbc92.npy|' \
    tmp/run_real_wbc_ex34.sh > tmp/run_real_wbc_ex35.sh
bash tmp/run_real_wbc_ex35.sh > tmp/log_real_wbc92.txt 2>&1
grep 'HDG\]\|VMC-T\|STAIRDBG' tmp/log_real_wbc92.txt | tail -22