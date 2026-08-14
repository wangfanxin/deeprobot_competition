#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc114.txt|tmp/log_real_wbc115.txt|' \
    -e 's|tmp/traj_wbc114.npy|tmp/traj_wbc115.npy|' \
    tmp/run_real_wbc_ex57.sh > tmp/run_real_wbc_ex58.sh
bash tmp/run_real_wbc_ex58.sh > tmp/log_real_wbc115.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc115.txt | tail -18