#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_YAW_DIFF=4.0/S10_FP_YAW_DIFF=2.0 S10_STAIR_HDG_K=0.6 S10_STAIR_HDG_D=4.0 S10_STAIR_HDG_OM=0.35/' \
    -e 's|tmp/log_real_wbc96.txt|tmp/log_real_wbc97.txt|' \
    -e 's|tmp/traj_wbc96.npy|tmp/traj_wbc97.npy|' \
    tmp/run_real_wbc_ex39.sh > tmp/run_real_wbc_ex40.sh
bash tmp/run_real_wbc_ex40.sh > tmp/log_real_wbc97.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc97.txt | tail -18