#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
/home/wfx/DR_competition/.venv/bin/python tmp/test_qpsolve2.py 2>&1 | tail -8
