#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc106.txt|tmp/log_real_wbc107.txt|' \
    -e 's|tmp/traj_wbc106.npy|tmp/traj_wbc107.npy|' \
    tmp/run_real_wbc_ex49.sh > tmp/run_real_wbc_ex50.sh
bash tmp/run_real_wbc_ex50.sh > tmp/log_real_wbc107.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc107.txt | tail -16