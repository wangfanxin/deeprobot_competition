#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc119.txt|tmp/log_real_wbc120.txt|' \
    -e 's|tmp/traj_wbc119.npy|tmp/traj_wbc120.npy|' \
    tmp/run_real_wbc_ex62.sh > tmp/run_real_wbc_ex63.sh
bash tmp/run_real_wbc_ex63.sh > tmp/log_real_wbc120.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc120.txt | tail -20