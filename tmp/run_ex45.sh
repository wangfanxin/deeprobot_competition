#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_HDG_K=1.0/S10_STAIR_HDG_K=1.5/' \
    -e 's/S10_STAIR_HDG_KI=0.4/S10_STAIR_HDG_KI=0.15/' \
    -e 's/S10_FP_YAW_KP=4.0/S10_FP_YAW_KP=12.0/' \
    -e 's|tmp/log_real_wbc101.txt|tmp/log_real_wbc102.txt|' \
    -e 's|tmp/traj_wbc101.npy|tmp/traj_wbc102.npy|' \
    tmp/run_real_wbc_ex44.sh > tmp/run_real_wbc_ex45.sh
bash tmp/run_real_wbc_ex45.sh > tmp/log_real_wbc102.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc102.txt | tail -16