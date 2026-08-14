#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_DRIVE_FLOOR=3.0/S10_FP_DRIVE_FLOOR=3.0 S10_FP_RECOV_DROP=0.10 S10_FP_RECOV_K=20/' \
    -e 's|tmp/log_real_wbc112.txt|tmp/log_real_wbc113.txt|' \
    -e 's|tmp/traj_wbc112.npy|tmp/traj_wbc113.npy|' \
    tmp/run_real_wbc_ex55.sh > tmp/run_real_wbc_ex56.sh
bash tmp/run_real_wbc_ex56.sh > tmp/log_real_wbc113.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc113.txt | tail -18