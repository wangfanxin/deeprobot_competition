#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_WIN_VX=1.0/S10_STAIR_WIN_VX=0.6/' \
    -e 's/S10_STAIR_CREST_VX=1.0/S10_STAIR_CREST_VX=0.6/' \
    -e 's|tmp/log_real_wbc107.txt|tmp/log_real_wbc108.txt|' \
    -e 's|tmp/traj_wbc107.npy|tmp/traj_wbc108.npy|' \
    tmp/run_real_wbc_ex50.sh > tmp/run_real_wbc_ex51.sh
bash tmp/run_real_wbc_ex51.sh > tmp/log_real_wbc108.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc108.txt | tail -16