#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed "s/S10_STAIR_EXEC_D=0.5/S10_STAIR_EXEC_D=0.3/" tmp/run_real_wbc_ex05.sh > tmp/run_real_wbc_ex03.sh
bash tmp/run_real_wbc_ex03.sh > tmp/log_real_wbc53.txt 2>&1; echo EXIT=$?