# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench_qp1.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_STAIR_SWING_D=0.30", "S10_STAIR_SWING_D=0.15")
src = src.replace("S10_STAIR_EXEC_D=0.5", "S10_STAIR_EXEC_D=1.2")
p.write_text(src, encoding="utf-8")
print(src)