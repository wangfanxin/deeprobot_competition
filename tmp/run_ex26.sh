#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc81.txt|tmp/log_real_wbc83.txt|' \
    -e 's|tmp/traj_wbc81.npy|tmp/traj_wbc83.npy|' \
    tmp/run_real_wbc_ex24.sh > tmp/run_real_wbc_ex26.sh
bash tmp/run_real_wbc_ex26.sh > tmp/log_real_wbc83.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc83.txt | tail -16