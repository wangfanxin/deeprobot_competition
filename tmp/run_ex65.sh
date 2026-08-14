#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc121.txt|tmp/log_real_wbc122.txt|' \
    -e 's|tmp/traj_wbc121.npy|tmp/traj_wbc122.npy|' \
    tmp/run_real_wbc_ex64.sh > tmp/run_real_wbc_ex65.sh
bash tmp/run_real_wbc_ex65.sh > tmp/log_real_wbc122.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc122.txt | tail -18