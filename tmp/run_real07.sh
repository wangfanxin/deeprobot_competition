#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
cp tmp/run_real_qp7.sh tmp/run_real_qp8.sh
sed -i -e "s/S10_STAIR_SWING_D=0.12/S10_STAIR_SWING_D=0.30/" \
       -e "s/S10_QP_DEBUG=2/S10_QP_DEBUG=2 S10_QP_SW_TGT_RATE=2.0/" \
    tmp/run_real_qp8.sh
bash tmp/run_real_qp8.sh > tmp/log_real07.txt 2>&1; echo EXIT=$?