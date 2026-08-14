#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc121.txt|tmp/log_real_wbc124.txt|' \
    -e 's|tmp/traj_wbc121.npy|tmp/traj_wbc124.npy|' \
    tmp/run_real_wbc_ex64.sh > tmp/run_real_wbc_ex67.sh
bash tmp/run_real_wbc_ex67.sh > tmp/log_real_wbc124.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc124.txt | tail -18