#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_WHEEL_K=4/S10_FP_WHEEL_K=12/' \
    -e 's|tmp/log_real_wbc94.txt|tmp/log_real_wbc95.txt|' \
    -e 's|tmp/traj_wbc94.npy|tmp/traj_wbc95.npy|' \
    tmp/run_real_wbc_ex37.sh > tmp/run_real_wbc_ex38.sh
bash tmp/run_real_wbc_ex38.sh > tmp/log_real_wbc95.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc95.txt | tail -16