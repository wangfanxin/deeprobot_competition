#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc113.txt|tmp/log_real_wbc114.txt|' \
    -e 's|tmp/traj_wbc113.npy|tmp/traj_wbc114.npy|' \
    tmp/run_real_wbc_ex56.sh > tmp/run_real_wbc_ex57.sh
bash tmp/run_real_wbc_ex57.sh > tmp/log_real_wbc114.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc114.txt | tail -18