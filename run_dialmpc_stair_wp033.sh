#!/bin/bash
# Rollback 2026-08-17: nav + BodyMPPI + CarVMC cruise, RL-stair, TK1/TK2.
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/s10_dial_mpc}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export S10_USE_VIEWER=0
export S10_INIT_YAW=1.5708
export S10_AUTO_MAX_WP=33 S10_TEST_MAX_SIM=600 S10_STUCK_TIMEOUT=90
export S10_NAV_HZ=20
export S10_VMC_TERRAIN=lidar S10_VMC_MODE=rlstair S10_VMC_USE_NAV=0
export S10_AUTO_VMAX=3.0 S10_AUTO_LOOKAHEAD=2.0 S10_AUTO_VLIM_LOOKAHEAD=3.0
export S10_AUTO_STAIR_VX=1.5
export S10_CTE_GAIN=4.0 S10_AUTO_CTE_MAX=2.0 S10_AUTO_CTE_ERR_GATE=1.0
export S10_RL_ELEV=1 S10_STAIR_ENTER_DIST=3.5
export S10_MPPI_OBSTACLE=1 S10_LIDAR_WALL=1
export VMC_MPPI_N=2048 VMC_MPPI_H=20
export S10_TK1=1 S10_TK2=1
export S10_TRAJ_FILE=tmp/cruise_vmc_stair_traj.csv
exec /home/wfx/DR_competition/.venv/bin/python src/S10_sdk_deploy/scripts/cruise_vmc_noros.py
