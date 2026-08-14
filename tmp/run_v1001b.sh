#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
cp tmp/run_real_v1000.sh tmp/run_real_v1001.sh
sed -i "s|export S10_QP_LAM_W=0.05 S10_QP_SW_FF=1.0 S10_QP_DEBUG=2|export S10_QP_LAM_W=0.05 S10_QP_SW_FF=1.0 S10_QP_DEBUG=2 S10_QP_FOLLOW_GAP=-0.005 S10_QP_FACE_DRIVE=1.0|" tmp/run_real_v1001.sh
grep -E "FOLLOW|FACE_DRIVE" tmp/run_real_v1001.sh
bash tmp/run_real_v1001.sh > tmp/log_real_v1001.txt 2>&1; echo EXIT=$?