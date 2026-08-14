#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc100.txt|tmp/log_real_wbc101.txt|' \
    -e 's|tmp/traj_wbc100.npy|tmp/traj_wbc101.npy|' \
    tmp/run_real_wbc_ex43.sh > tmp/run_real_wbc_ex44.sh
bash tmp/run_real_wbc_ex44.sh > tmp/log_real_wbc101.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc101.txt | tail -18