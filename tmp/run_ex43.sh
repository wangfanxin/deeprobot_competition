#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_DRIVE_FLOOR=3.0/S10_FP_DRIVE_FLOOR=3.0 S10_FP_KP_SW=120/' \
    -e 's|tmp/log_real_wbc99.txt|tmp/log_real_wbc100.txt|' \
    -e 's|tmp/traj_wbc99.npy|tmp/traj_wbc100.npy|' \
    tmp/run_real_wbc_ex42.sh > tmp/run_real_wbc_ex43.sh
bash tmp/run_real_wbc_ex43.sh > tmp/log_real_wbc100.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc100.txt | tail -16