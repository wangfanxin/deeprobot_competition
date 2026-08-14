#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
cp tmp/run_real_qp8.sh tmp/run_real_qp9.sh
sed -i -e "s/S10_STAIR_WIN_VX=3.2/S10_STAIR_WIN_VX=2.5/" \
       -e "s/S10_STAIR_CREST_VX=3.2/S10_STAIR_CREST_VX=2.5/" \
    tmp/run_real_qp9.sh
bash tmp/run_real_qp9.sh > tmp/log_real09.txt 2>&1; echo EXIT=$?