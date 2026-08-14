#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e "s/S10_STAIR_WIN_VX=2.0/S10_STAIR_WIN_VX=3.2/" \
    -e "s/S10_STAIR_CREST_VX=1.5/S10_STAIR_CREST_VX=3.2/" \
    tmp/run_real_qp4.sh > tmp/run_real_qp5.sh
bash tmp/run_real_qp5.sh > tmp/log_real03.txt 2>&1; echo EXIT=$?