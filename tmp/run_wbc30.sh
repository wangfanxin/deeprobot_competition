#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
bash tmp/run_real_wbc.sh > tmp/log_real_wbc30.txt 2>&1
grep -n "VMC-T" tmp/log_real_wbc30.txt | tail -4
echo "=== max y ==="
grep "VMC-T" tmp/log_real_wbc30.txt | sed -E "s/.*pos=\((-?[0-9.]+),(-?[0-9.]+),.*/\1 \2/" | awk "{if (\$2+0 > m) m=\$2+0} END {print m}"
tail -3 tmp/log_real_wbc30.txt