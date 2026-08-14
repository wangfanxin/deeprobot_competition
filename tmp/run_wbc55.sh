#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_STAIR_DEBUG=1
bash tmp/run_real_wbc_ex03.sh > tmp/log_real_wbc55.txt 2>&1
grep "STAIRDBG" tmp/log_real_wbc55.txt | grep -E "38\.[0-9]" | head -8
echo "=== result ==="
tail -3 tmp/log_real_wbc55.txt