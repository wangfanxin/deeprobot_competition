#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
for tk in 0.5 0.7 0.8; do
  S10_GLOBAL_TANGENT_K=$tk S10_REF_DUMP=tmp/ref_path_tk${tk}.npz S10_DUMP_ONLY=1 bash tmp/run_v780.sh >/dev/null 2>&1
  echo "dump tk=$tk rc=$?"
done
/home/wfx/DR_competition/.venv/bin/python tmp/analyze_radius.py