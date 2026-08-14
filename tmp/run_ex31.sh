#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_SWING_D=0.15/S10_STAIR_SWING_D=0.20/' \
    -e 's/S10_FP_BODY_KD=0.4 S10_FP_BODY_K=0.2 S10_FP_BODY_PD=0.2/S10_FP_BODY_KD=0.4 S10_FP_BODY_K=0.2 S10_FP_BODY_PD=0.2 S10_FP_YAW_DIFF=4.0 S10_FP_YAW_KP=4.0/' \
    -e 's|tmp/log_real_wbc87.txt|tmp/log_real_wbc88.txt|' \
    -e 's|tmp/traj_wbc87.npy|tmp/traj_wbc88.npy|' \
    tmp/run_real_wbc_ex30.sh > tmp/run_real_wbc_ex31.sh
bash tmp/run_real_wbc_ex31.sh > tmp/log_real_wbc88.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc88.txt | tail -18