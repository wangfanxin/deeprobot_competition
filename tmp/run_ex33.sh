#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's|tmp/log_real_wbc88.txt|tmp/log_real_wbc90.txt|' \
    -e 's|tmp/traj_wbc88.npy|tmp/traj_wbc90.npy|' \
    tmp/run_real_wbc_ex31.sh > tmp/run_real_wbc_ex33.sh
bash tmp/run_real_wbc_ex33.sh > tmp/log_real_wbc90.txt 2>&1
grep 'VMC-T\|STAIRDBG' tmp/log_real_wbc90.txt | tail -18