#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc93.txt|tmp/log_real_wbc96.txt|' \
    -e 's|tmp/traj_wbc93.npy|tmp/traj_wbc96.npy|' \
    tmp/run_real_wbc_ex36.sh > tmp/run_real_wbc_ex39.sh
bash tmp/run_real_wbc_ex39.sh > tmp/log_real_wbc96.txt 2>&1
grep 'VMC-T\|STAIRDBG\|HDG\]' tmp/log_real_wbc96.txt | tail -20