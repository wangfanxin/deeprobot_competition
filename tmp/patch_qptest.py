# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = "    def _qp_solve(self, body, wheel_xyz, stance_mask, lam_ref):\n        if self._osqp is None:\n            return lam_ref"
assert old in src
new = "    def _qp_solve(self, body, wheel_xyz, stance_mask, lam_ref):\n        print('QPTEST enter _qp_solve osqp=%s' % (self._osqp is not None), flush=True)\n        if self._osqp is None:\n            return lam_ref"
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")