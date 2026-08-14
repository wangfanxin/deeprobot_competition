#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
timeout 160 bash tmp/run_bench13.sh > tmp/log_bench13_now.txt 2>&1
echo EXIT=True
