#!/bin/bash
# DIAL-MPC + built-in stair contact planner (stair_dial_noros.py)
# Correct model: DIAL-MPC S10WheeledEnv + StairContactPlanner + StairStanceGuard.
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/s10_dial_mpc}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export S10_USE_VIEWER=0
export S10_AUTO_MAX_WP=33 S10_TEST_MAX_SIM=600 S10_STUCK_TIMEOUT=90
export S10_NAV_HZ=20
export S10_AUTO_VMAX=4.0 S10_AUTO_LOOKAHEAD=3.5 S10_AUTO_VLIM_LOOKAHEAD=3.0
export S10_AUTO_CTE_MAX=1.5 S10_AUTO_CTE_ERR_GATE=1.0 S10_AUTO_ARRIVE_ERR=0.5
export S10_MPC_YAML=/home/wfx/DR_competition/0810new/deeprobot_competition/doc/s10_mpc_deploy.yaml
export S10_MPC_NSAMPLE=512 S10_MPC_HSAMPLE=14 S10_MPC_NDIFFUSE=1
export S10_MPC_VEL_SCALE=56 S10_MPC_WHEEL_TAU_SCALE=3.0 S10_MPC_KP=80 S10_MPC_KD=2.0 S10_MPC_HEIGHT_WEIGHT=0.1
export S10_STAND_TIME=0.6 S10_STAND_KP=120 S10_STAND_KD=3.0
export S10_TRAJ_FILE=tmp/dialmpc_stair_traj.csv
exec /home/wfx/DR_competition/.venv/bin/python src/S10_sdk_deploy/scripts/stair_dial_noros.py
