#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
/home/wfx/DR_competition/.venv/bin/python tmp/test_qpsolve.py > tmp/qpsolve_out.txt 2>&1
grep -e 'QP' tmp/qpsolve_out.txt
