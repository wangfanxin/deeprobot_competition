#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_YAW_DIFF=10.0/S10_FP_YAW_DIFF=0.0/' \
    -e 's/S10_FP_YAW_ERR_K=10.0/S10_FP_YAW_ERR_K=0.0/' \
    -e 's|tmp/log_real_wbc110.txt|tmp/log_real_wbc111.txt|' \
    -e 's|tmp/traj_wbc110.npy|tmp/traj_wbc111.npy|' \
    tmp/run_real_wbc_ex53.sh > tmp/run_real_wbc_ex54.sh
bash tmp/run_real_wbc_ex54.sh > tmp/log_real_wbc111.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc111.txt | tail -18