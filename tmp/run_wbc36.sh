#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e "s/S10_STAIR_WIN_VX=2.2/S10_STAIR_WIN_VX=1.8/" \
    -e "s/S10_STAIR_CREST_VX=2.2/S10_STAIR_CREST_VX=1.8/" \
    tmp/run_real_wbc_r.sh > tmp/run_real_wbc_r18.sh
bash tmp/run_real_wbc_r18.sh > tmp/log_real_wbc36.txt 2>&1; echo EXIT=$?