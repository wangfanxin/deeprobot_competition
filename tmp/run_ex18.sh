#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_WIN_VX=1.8/S10_STAIR_WIN_VX=1.2/' \
    -e 's/S10_STAIR_CREST_VX=1.8/S10_STAIR_CREST_VX=1.2/' \
    -e 's/S10_FP_STAND_DROP=0.22/S10_FP_STAND_DROP=0.22 S10_FP_DRIVE_FLOOR=3.0/' \
    -e 's|tmp/log_real_wbc73.txt|tmp/log_real_wbc75.txt|' \
    -e 's|tmp/traj_wbc73.npy|tmp/traj_wbc75.npy|' \
    tmp/run_real_wbc_ex17.sh > tmp/run_real_wbc_ex18.sh
bash tmp/run_real_wbc_ex18.sh > tmp/log_real_wbc75.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc75.txt | tail -14