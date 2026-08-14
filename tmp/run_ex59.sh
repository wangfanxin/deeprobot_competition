#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc115.txt|tmp/log_real_wbc116.txt|' \
    -e 's|tmp/traj_wbc115.npy|tmp/traj_wbc116.npy|' \
    tmp/run_real_wbc_ex58.sh > tmp/run_real_wbc_ex59.sh
bash tmp/run_real_wbc_ex59.sh > tmp/log_real_wbc116.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc116.txt | tail -20