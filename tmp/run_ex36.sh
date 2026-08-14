#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_FP_YAW_DIFF=4.0/S10_FP_YAW_DIFF=4.0 S10_STAIR_OM_SCALE=1.0/' \
    -e 's|tmp/log_real_wbc92.txt|tmp/log_real_wbc93.txt|' \
    -e 's|tmp/traj_wbc92.npy|tmp/traj_wbc93.npy|' \
    tmp/run_real_wbc_ex35.sh > tmp/run_real_wbc_ex36.sh
bash tmp/run_real_wbc_ex36.sh > tmp/log_real_wbc93.txt 2>&1
grep 'HDG\]\|VMC-T\|STAIRDBG' tmp/log_real_wbc93.txt | tail -24