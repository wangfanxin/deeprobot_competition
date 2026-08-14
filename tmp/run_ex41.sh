#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc97.txt|tmp/log_real_wbc98.txt|' \
    -e 's|tmp/traj_wbc97.npy|tmp/traj_wbc98.npy|' \
    tmp/run_real_wbc_ex40.sh > tmp/run_real_wbc_ex41.sh
bash tmp/run_real_wbc_ex41.sh > tmp/log_real_wbc98.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc98.txt | tail -18