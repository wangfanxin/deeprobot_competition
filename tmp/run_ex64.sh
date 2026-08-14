#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_SWING_D=0.35/S10_STAIR_SWING_D=0.45/' \
    -e 's/S10_FP_KP_SW=220/S10_FP_KP_SW=160/' \
    -e 's|tmp/log_real_wbc120.txt|tmp/log_real_wbc121.txt|' \
    -e 's|tmp/traj_wbc120.npy|tmp/traj_wbc121.npy|' \
    tmp/run_real_wbc_ex63.sh > tmp/run_real_wbc_ex64.sh
bash tmp/run_real_wbc_ex64.sh > tmp/log_real_wbc121.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc121.txt | tail -18