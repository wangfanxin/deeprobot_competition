#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_STAND_DROP=0.22/S10_FP_STAND_DROP=0.22 S10_WHEEL_PRESS=0.05/' \
    -e 's|tmp/log_real_wbc75.txt|tmp/log_real_wbc77.txt|' \
    -e 's|tmp/traj_wbc75.npy|tmp/traj_wbc77.npy|' \
    tmp/run_real_wbc_ex18.sh > tmp/run_real_wbc_ex20.sh
bash tmp/run_real_wbc_ex20.sh > tmp/log_real_wbc77.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc77.txt | tail -16