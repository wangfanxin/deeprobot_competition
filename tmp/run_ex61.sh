#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_DRIVE_FLOOR=8.0/S10_FP_DRIVE_FLOOR=5.0/' \
    -e 's|tmp/log_real_wbc117.txt|tmp/log_real_wbc118.txt|' \
    -e 's|tmp/traj_wbc117.npy|tmp/traj_wbc118.npy|' \
    tmp/run_real_wbc_ex60.sh > tmp/run_real_wbc_ex61.sh
bash tmp/run_real_wbc_ex61.sh > tmp/log_real_wbc118.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc118.txt | tail -18