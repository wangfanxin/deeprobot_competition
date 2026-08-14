#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_SWING_D=0.30/S10_STAIR_SWING_D=0.15/' \
    -e 's|tmp/log_real_wbc86.txt|tmp/log_real_wbc87.txt|' \
    -e 's|tmp/traj_wbc86.npy|tmp/traj_wbc87.npy|' \
    tmp/run_real_wbc_ex29.sh > tmp/run_real_wbc_ex30.sh
bash tmp/run_real_wbc_ex30.sh > tmp/log_real_wbc87.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc87.txt | tail -16