#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_SWING_D=0.20/S10_STAIR_SWING_D=0.30/' \
    -e 's/S10_STAIR_WIN_VX=1.2/S10_STAIR_WIN_VX=1.0/' \
    -e 's/S10_STAIR_CREST_VX=1.2/S10_STAIR_CREST_VX=1.0/' \
    -e 's|tmp/log_real_wbc90.txt|tmp/log_real_wbc91.txt|' \
    -e 's|tmp/traj_wbc90.npy|tmp/traj_wbc91.npy|' \
    tmp/run_real_wbc_ex33.sh > tmp/run_real_wbc_ex34.sh
bash tmp/run_real_wbc_ex34.sh > tmp/log_real_wbc91.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc91.txt | tail -18