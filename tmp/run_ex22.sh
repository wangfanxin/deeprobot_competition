#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_KP_POS=220/S10_FP_KP_POS=120/' \
    -e 's|tmp/log_real_wbc78.txt|tmp/log_real_wbc79.txt|' \
    -e 's|tmp/traj_wbc78.npy|tmp/traj_wbc79.npy|' \
    tmp/run_real_wbc_ex21.sh > tmp/run_real_wbc_ex22.sh
bash tmp/run_real_wbc_ex22.sh > tmp/log_real_wbc79.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc79.txt | tail -18