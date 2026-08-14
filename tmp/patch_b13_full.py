# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/run_bench13.sh")
src = p.read_text(encoding="utf-8")
old = 'timeout 150 bash tmp/run_stw_smoke.sh 2>&1 | grep -E "BENCH|MODE|VMC-T|卡死|侧翻|完成" | head -30'
assert old in src
new = "exec bash tmp/run_stw_smoke.sh"
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")