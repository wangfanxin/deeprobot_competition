#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_STAIR_DEBUG=1
bash tmp/run_real_wbc.sh > tmp/log_real_wbc27.txt 2>&1; echo EXIT=$?