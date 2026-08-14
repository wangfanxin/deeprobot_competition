#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
grep RISER tmp/run_real_qp5.sh
bash tmp/run_real_qp5.sh > tmp/log_real04.txt 2>&1; echo EXIT=$?