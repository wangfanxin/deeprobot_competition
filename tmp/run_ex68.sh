#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_DRIVE_FLOOR=5.0/S10_FP_DRIVE_FLOOR=5.0 S10_STAIR_LIFT_MARGIN=0.0/' \
    -e 's|tmp/log_real_wbc124.txt|tmp/log_real_wbc125.txt|' \
    -e 's|tmp/traj_wbc124.npy|tmp/traj_wbc125.npy|' \
    tmp/run_real_wbc_ex67.sh > tmp/run_real_wbc_ex68.sh
bash tmp/run_real_wbc_ex68.sh > tmp/log_real_wbc125.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc125.txt | tail -18