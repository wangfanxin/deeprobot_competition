#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
sed -e 's/S10_STAIR_HDG_K=1.5/S10_STAIR_HDG_K=1.0/' \
    -e 's/S10_NMPC_KP_YAW=10.0/S10_NMPC_KP_YAW=6.0/' \
    -e 's/S10_NMPC_YAW_ERR_K=2.0/S10_NMPC_YAW_ERR_K=1.5/' \
    -e 's|tmp/log_nmpc_real4.txt|tmp/log_nmpc_real5.txt|' \
    tmp/run_nmpc_real4.sh > tmp/run_nmpc_real5.sh
bash tmp/run_nmpc_real5.sh > tmp/log_nmpc_real5.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_real5.txt | tail -12