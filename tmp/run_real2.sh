#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
bash tmp/run_real1.sh > tmp/run_real2.out 2>&1
grep -e 'VMC-T' -e '侧翻' -e '卡死' -e '完成' -e '到达' -e 'MODE' tmp/log_real_stair1.txt | tail -40
