#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc77.txt|tmp/log_real_wbc78.txt|' \
    -e 's|tmp/traj_wbc77.npy|tmp/traj_wbc78.npy|' \
    tmp/run_real_wbc_ex20.sh > tmp/run_real_wbc_ex21.sh
bash tmp/run_real_wbc_ex21.sh > tmp/log_real_wbc78.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc78.txt | tail -20