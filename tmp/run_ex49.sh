#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc105.txt|tmp/log_real_wbc106.txt|' \
    -e 's|tmp/traj_wbc105.npy|tmp/traj_wbc106.npy|' \
    tmp/run_real_wbc_ex48.sh > tmp/run_real_wbc_ex49.sh
bash tmp/run_real_wbc_ex49.sh > tmp/log_real_wbc106.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc106.txt | tail -18