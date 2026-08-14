#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_BODY_KD=0.4 S10_FP_BODY_K=0.2 S10_FP_BODY_PD=0.2 S10_FP_YAW_KP=6.0 S10_FP_YAW_DIFF=6.0/S10_FP_BODY_KD=0.4 S10_FP_BODY_K=0.2 S10_FP_BODY_PD=0.2/' \
    -e 's|tmp/log_real_wbc85.txt|tmp/log_real_wbc86.txt|' \
    -e 's|tmp/traj_wbc85.npy|tmp/traj_wbc86.npy|' \
    tmp/run_real_wbc_ex28.sh > tmp/run_real_wbc_ex29.sh
bash tmp/run_real_wbc_ex29.sh > tmp/log_real_wbc86.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc86.txt | tail -16