#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
grep -n "SWING_TO\|_to =" src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py | head -3
cp tmp/run_real_v1001.sh tmp/run_real_pogo.sh
sed -i "s|S10_QP_LAM_W=0.05 S10_QP_SW_FF=1.0 S10_QP_DEBUG=3 S10_QP_SW_TGT_RATE=2.0|S10_QP_LAM_W=0.05 S10_QP_SW_FF=1.0 S10_QP_DEBUG=0 S10_QP_SW_TGT_RATE=2.0 S10_STAIR_SWING_TO=0.4|" tmp/run_real_pogo.sh
grep -E "SWING_TO|DEBUG" tmp/run_real_pogo.sh
bash tmp/run_real_pogo.sh > tmp/log_real_pogo01.txt 2>&1; echo EXIT=$?