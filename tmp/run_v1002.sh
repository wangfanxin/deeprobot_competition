#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_QP_FOLLOW_GAP=-0.005
export S10_QP_FACE_DRIVE=1.0
bash tmp/run_real_v1000.sh > tmp/log_real_v1002.txt 2>&1; echo EXIT=$?