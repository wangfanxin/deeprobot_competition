#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -i 's/S10_FP_STAND_DROP=0.22/S10_FP_STAND_DROP=0.22 S10_NAV_DEBUG=1/' tmp/run_real_wbc_ex16.sh
grep -n 'NAV_DEBUG' tmp/run_real_wbc_ex16.sh
bash tmp/run_real_wbc_ex16.sh > tmp/log_real_wbc74.txt 2>&1
grep 'NAV]' tmp/log_real_wbc74.txt | tail -32