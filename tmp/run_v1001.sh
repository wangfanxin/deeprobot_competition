#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
cp tmp/run_real_v1000.sh tmp/run_real_v1001.sh
sed -i -e "s/S10_QP_FOLLOW_GAP=0.003/S10_QP_FOLLOW_GAP=-0.005/" \
       -e "s/S10_QP_FACE_DRIVE=2.0/S10_QP_FACE_DRIVE=1.0/" \
    tmp/run_real_v1001.sh
grep -E "FOLLOW|FACE_DRIVE" tmp/run_real_v1001.sh
bash tmp/run_real_v1001.sh > tmp/log_real_v1001.txt 2>&1; echo EXIT=$?