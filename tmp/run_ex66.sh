#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc122.txt|tmp/log_real_wbc123.txt|' \
    -e 's|tmp/traj_wbc122.npy|tmp/traj_wbc123.npy|' \
    tmp/run_real_wbc_ex65.sh > tmp/run_real_wbc_ex66.sh
bash tmp/run_real_wbc_ex66.sh > tmp/log_real_wbc123.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc123.txt | tail -18