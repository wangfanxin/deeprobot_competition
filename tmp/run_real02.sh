#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e "s/S10_STAIR_VX_RAMP=4.0/S10_STAIR_VX_RAMP=10.0/" \
    -e "s/S10_STAIR_WIN_VX=1.8/S10_STAIR_WIN_VX=2.0/" \
    tmp/run_real_qp3.sh > tmp/run_real_qp4.sh
bash tmp/run_real_qp4.sh > tmp/log_real02.txt 2>&1; echo EXIT=$?