#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_WIN_VX=0.6/S10_STAIR_WIN_VX=1.0/' \
    -e 's/S10_STAIR_CREST_VX=0.6/S10_STAIR_CREST_VX=1.0/' \
    -e 's|tmp/log_real_wbc109.txt|tmp/log_real_wbc110.txt|' \
    -e 's|tmp/traj_wbc109.npy|tmp/traj_wbc110.npy|' \
    tmp/run_real_wbc_ex52.sh > tmp/run_real_wbc_ex53.sh
bash tmp/run_real_wbc_ex53.sh > tmp/log_real_wbc110.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc110.txt | tail -18