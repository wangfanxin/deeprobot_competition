#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
cp tmp/run_real_press.sh tmp/run_real_press3.sh
sed -i -e "s/S10_STAIR_WIN_VX=2.5/S10_STAIR_WIN_VX=2.2/" \
       -e "s/S10_STAIR_CREST_VX=2.5/S10_STAIR_CREST_VX=2.2/" \
       -e "s/S10_QP_K_OVER=1500/S10_QP_K_OVER=1200/" \
    tmp/run_real_press3.sh
bash tmp/run_real_press3.sh > tmp/log_real_press03.txt 2>&1; echo EXIT=$?