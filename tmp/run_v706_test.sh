#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_AUTO_MAX_WP=6 S10_TEST_MAX_SIM=120 S10_USE_VIEWER=0 S10_MPC_ENABLE=1 S10_MODE=auto_nav S10_LIDAR_BACKEND=cpu
/home/wfx/DR_competition/.venv/bin/python src/S10_sdk_deploy/scripts/cruise_noros.py > tmp/run_v706_dialmpc_default.log 2>&1
echo "EXIT=$?"
tail -n 8 tmp/run_v706_dialmpc_default.log
