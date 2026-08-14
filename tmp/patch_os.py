#!/usr/bin/env python3
import io
path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()
old = '''"""
import numpy as np'''
new = '''"""
import os

import numpy as np'''
assert old in src, "anchor"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched")