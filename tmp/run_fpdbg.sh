#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_FP_DEBUG=1 S10_TEST_MAX_SIM=12 S10_STUCK_TIMEOUT=15
bash tmp/run_bench9full.sh > tmp/log_fpdbg.txt 2>&1
grep -e 'FP]' -e 'VMC-T' tmp/log_fpdbg.txt | head -80
