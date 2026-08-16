#!/bin/bash
# 组合技能(890 cruise + rl-stair) wp0-33 带头视图运行
# 用法: bash run_wp033_viewer.sh   (在 WSL 内运行, 需要显示器/WSLg)
cd /home/wfx/DR_competition/0810new/deeprobot_competition
env \
  S10_USE_VIEWER=1 S10_INIT_YAW=1.5708 \
  S10_AUTO_MAX_WP=33 S10_TEST_MAX_SIM=600 S10_STUCK_TIMEOUT=90 \
  S10_VMC_TERRAIN=lidar S10_VMC_MODE=rlstair S10_VMC_USE_NAV=1 \
  S10_AUTO_VMAX=3.5 S10_AUTO_LOOKAHEAD=3.5 S10_AUTO_VLIM_LOOKAHEAD=3.5 \
  S10_AUTO_STAIR_VX=1.5 S10_ELEV_KNOWN_RAMP=2.0 \
  S10_RL_ELEV=1 S10_PRETRANS=1 S10_PRETRANS_Y0=32.0 \
  S10_PRETRANS_EXIT_Y0=40.4 S10_PRETRANS_EXIT_LEN=2.0 \
  S10_STAIR_ENTER_DIST=3.5 S10_RL_WARMUP=0 S10_STAIR_EXIT_VX=1.5 \
  /home/wfx/DR_competition/.venv/bin/python src/S10_sdk_deploy/scripts/cruise_vmc_noros.py