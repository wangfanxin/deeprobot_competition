# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
old = "export S10_YAW_DAMP=2.0"
assert old in src
new = "export S10_YAW_DAMP=2.0\nexport S10_VMC_TERRAIN_KIN=1"
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")