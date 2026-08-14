#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
cp tmp/run_real_qp9.sh tmp/run_real_press.sh
sed -i -e "s/S10_QP_KP_SW=70/S10_QP_KP_SW=30/" \
       -e "s/S10_QP_KD_SW=50/S10_QP_KD_SW=80/" \
       -e "s/S10_QP_K_OVER=800/S10_QP_K_OVER=1500/" \
       -e "s/S10_QP_DEBUG=2/S10_QP_DEBUG=3/" \
    tmp/run_real_press.sh
bash tmp/run_real_press.sh > tmp/log_real_press01.txt 2>&1; echo EXIT=$?