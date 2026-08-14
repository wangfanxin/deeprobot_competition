#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_KP_SW=120/S10_FP_KP_SW=220/' \
    -e 's|tmp/log_real_wbc118.txt|tmp/log_real_wbc119.txt|' \
    -e 's|tmp/traj_wbc118.npy|tmp/traj_wbc119.npy|' \
    tmp/run_real_wbc_ex61.sh > tmp/run_real_wbc_ex62.sh
bash tmp/run_real_wbc_ex62.sh > tmp/log_real_wbc119.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc119.txt | tail -20