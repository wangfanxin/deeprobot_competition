# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                if not _lead:
                    _roll_gate = abs(getattr(self, '_body_roll', 0.0)) < 0.08"""
assert old in src
new = """                if not _lead:
                    # v954: 门控放宽到 0.15——FL 上台面后 body 几何 roll 约
                    # 0.1（腿吸收后观测 -0.1），0.08 太紧 FR 永远等不到
                    _roll_gate = abs(getattr(self, '_body_roll', 0.0)) < 0.15"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v954")