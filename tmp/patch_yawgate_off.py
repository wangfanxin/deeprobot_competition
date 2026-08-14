# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
if "S10_STAIR_YAW_GATE" not in src:
    src = src.replace("export S10_YAW_DAMP=2.0", "export S10_YAW_DAMP=2.0\nexport S10_STAIR_YAW_GATE=1.0", 1)
p.write_text(src, encoding="utf-8")
print("patched yaw gate off for bench")