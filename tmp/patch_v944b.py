# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            _z_ref = 0.78
            _any_sw = float(np.max(step_lift)) > 0.5
            if _any_sw:
                _z_ref = float(body["pos"][2])"""
assert old in src
new = """            _z_ref = 0.78
            if float(getattr(self, '_any_swing', 0.0)) > 0.5:
                _z_ref = float(body["pos"][2])"""
src = src.replace(old, new, 1)
# 在 compute_tau 存 _any_swing
old2 = """        self._rear_swing = float(np.max(step_lift[2:4])) > 0.5"""
assert old2 in src
new2 = """        self._rear_swing = float(np.max(step_lift[2:4])) > 0.5
        self._any_swing = float(np.max(step_lift)) > 0.5"""
src = src.replace(old2, new2, 1)
p.write_text(src, encoding="utf-8")
print("patched v944b")