#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
for i in 1 2 3; do
  bash tmp/run_real_wbc_r18.sh > tmp/log_real_wbc4${i}.txt 2>&1
  y=$(grep "VMC-T" tmp/log_real_wbc4${i}.txt | sed -E "s/.*pos=\((-?[0-9.]+),(-?[0-9.]+),.*/\2/" | awk "{if (\$1+0 > m) m=\$1+0} END {print m}")
  echo "run $i: max_y=$y result=$(tail -2 tmp/log_real_wbc4${i}.txt | head -1)"
done