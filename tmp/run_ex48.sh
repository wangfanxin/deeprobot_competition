#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc104.txt|tmp/log_real_wbc105.txt|' \
    -e 's|tmp/traj_wbc104.npy|tmp/traj_wbc105.npy|' \
    tmp/run_real_wbc_ex47.sh > tmp/run_real_wbc_ex48.sh
bash tmp/run_real_wbc_ex48.sh > tmp/log_real_wbc105.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc105.txt | tail -18