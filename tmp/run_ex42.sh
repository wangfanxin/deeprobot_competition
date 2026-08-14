#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_HDG_K=0.6/S10_STAIR_HDG_K=1.0/' \
    -e 's/S10_STAIR_HDG_D=4.0/S10_STAIR_HDG_D=3.0 S10_STAIR_HDG_KI=0.4 S10_STAIR_HDG_OM=0.5/' \
    -e 's|tmp/log_real_wbc98.txt|tmp/log_real_wbc99.txt|' \
    -e 's|tmp/traj_wbc98.npy|tmp/traj_wbc99.npy|' \
    tmp/run_real_wbc_ex41.sh > tmp/run_real_wbc_ex42.sh
bash tmp/run_real_wbc_ex42.sh > tmp/log_real_wbc99.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc99.txt | tail -16