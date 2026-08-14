#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
/home/wfx/DR_competition/.venv/bin/python -m py_compile src/S10_sdk_deploy/s10_mpc/stair_wbc.py src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py && echo OK
export S10_TEST_MAX_SIM=50 S10_STUCK_TIMEOUT=22
bash tmp/run_bench_v901xml.sh > tmp/log_b901b.txt 2>&1
echo EXIT=True
