#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_STAND_DROP=0.22/S10_FP_STAND_DROP=0.22 S10_FP_BODY_KD=0.2/' \
    -e 's|tmp/log_real_wbc79.txt|tmp/log_real_wbc81.txt|' \
    -e 's|tmp/traj_wbc79.npy|tmp/traj_wbc81.npy|' \
    tmp/run_real_wbc_ex22.sh > tmp/run_real_wbc_ex24.sh
bash tmp/run_real_wbc_ex24.sh > tmp/log_real_wbc81.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc81.txt | tail -18