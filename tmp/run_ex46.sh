#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc102.txt|tmp/log_real_wbc103.txt|' \
    -e 's|tmp/traj_wbc102.npy|tmp/traj_wbc103.npy|' \
    tmp/run_real_wbc_ex45.sh > tmp/run_real_wbc_ex46.sh
bash tmp/run_real_wbc_ex46.sh > tmp/log_real_wbc103.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc103.txt | tail -18