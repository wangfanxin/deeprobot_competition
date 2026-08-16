#!/bin/bash
# DIAL-MPC full-body cruise wp0-33 (work-in-progress)
# Correct model: cruise_noros.py -> MPCController(S10WheeledEnv+MBDPI), NOT CarVMC.
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/s10_dial_mpc}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export S10_USE_VIEWER=0
export S10_AUTO_MAX_WP=33 S10_TEST_MAX_SIM=600 S10_STUCK_TIMEOUT=90
export S10_NAV_HZ=20
export S10_AUTO_VMAX=2.0 S10_AUTO_LOOKAHEAD=3.5 S10_AUTO_VLIM_LOOKAHEAD=3.0
export S10_AUTO_CTE_MAX=1.5 S10_AUTO_CTE_ERR_GATE=1.0 S10_AUTO_ARRIVE_ERR=0.5
export S10_START_CORRIDOR_X=0.5
export S10_MPPI_A_MAX=3.0 S10_MPPI_MU=0.8 S10_MPPI_OMAX=4.0 S10_MPPI_W_GUIDE=1.0 S10_MPPI_W_DIST=0.8
export S10_MPC_YAML=/home/wfx/DR_competition/0810new/deeprobot_competition/doc/s10_mpc_deploy.yaml
export S10_MPC_NSAMPLE=1024 S10_MPC_HSAMPLE=14 S10_MPC_NDIFFUSE=1
export S10_MPC_VEL_SCALE=30 S10_MPC_WHEEL_TAU_SCALE=2.0 S10_MPC_KP=60 S10_MPC_KD=1.5
export S10_STAND_TIME=0.6 S10_STAND_KP=120 S10_STAND_KD=3.0
export S10_TRAJ_FILE=tmp/dialmpc_traj.csv
export S10_MPC_DT=0.02
exec /home/wfx/DR_competition/.venv/bin/python src/S10_sdk_deploy/scripts/cruise_noros.py
