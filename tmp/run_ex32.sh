#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_YAW_DIFF=4.0/S10_FP_YAW_DIFF=4.0 S10_STAIR_OM_SCALE=1.0/' \
    -e 's|tmp/log_real_wbc88.txt|tmp/log_real_wbc89.txt|' \
    -e 's|tmp/traj_wbc88.npy|tmp/traj_wbc89.npy|' \
    tmp/run_real_wbc_ex31.sh > tmp/run_real_wbc_ex32.sh
bash tmp/run_real_wbc_ex32.sh > tmp/log_real_wbc89.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc89.txt | tail -16