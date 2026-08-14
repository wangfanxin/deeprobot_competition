#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e "s/S10_STAIR_WIN_VX=1.8/S10_STAIR_WIN_VX=3.5/" \
    -e "s/S10_STAIR_CREST_VX=1.8/S10_STAIR_CREST_VX=3.5/" \
    tmp/run_real_wbc_r18.sh > tmp/run_real_wbc_v35.sh
bash tmp/run_real_wbc_v35.sh > tmp/log_real_wbc49.txt 2>&1; echo EXIT=$?