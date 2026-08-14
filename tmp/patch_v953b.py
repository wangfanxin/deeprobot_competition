# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old2 = """        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)"""
assert old2 in src
new2 = """        self._body_roll = body["roll"]
        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)"""
src = src.replace(old2, new2, 1)
old3 = """                if not _lead:
                    _roll_gate = abs(body["roll"]) < 0.08"""
assert old3 in src
new3 = """                if not _lead:
                    _roll_gate = abs(getattr(self, '_body_roll', 0.0)) < 0.08"""
src = src.replace(old3, new3, 1)
p.write_text(src, encoding="utf-8")
print("patched v953b")