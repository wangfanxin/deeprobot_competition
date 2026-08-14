#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc83.txt|tmp/log_real_wbc84.txt|' \
    -e 's|tmp/traj_wbc83.npy|tmp/traj_wbc84.npy|' \
    tmp/run_real_wbc_ex26.sh > tmp/run_real_wbc_ex27.sh
bash tmp/run_real_wbc_ex27.sh > tmp/log_real_wbc84.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc84.txt | tail -18