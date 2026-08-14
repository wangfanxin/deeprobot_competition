#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e "s/S10_STAIR_WIN_VX=2.2/S10_STAIR_WIN_VX=3.0/" \
    -e "s/S10_STAIR_CREST_VX=2.2/S10_STAIR_CREST_VX=3.0/" \
    tmp/run_real_wbc.sh > tmp/run_real_wbc_v3.sh
bash tmp/run_real_wbc_v3.sh > tmp/log_real_wbc17.txt 2>&1; echo EXIT=$?