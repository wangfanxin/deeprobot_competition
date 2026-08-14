# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_real_stair.sh")
src = p.read_text(encoding="utf-8")
src = src.replace("S10_AUTO_LOOKAHEAD_STAIR=1.5 S10_AUTO_CTE_GAIN_STAIR=5.0 S10_AUTO_YAW_GAIN_STAIR=4.0", "S10_AUTO_LOOKAHEAD_STAIR=2.5 S10_AUTO_CTE_GAIN_STAIR=2.5 S10_AUTO_YAW_GAIN_STAIR=2.0")
p.write_text(src, encoding="utf-8")
print("patched")