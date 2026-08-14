# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                _ok = (_win_lo < d[i] < _win_hi
                       and (self._done[i] is False)
                       and (not _lead or _front_done)
                       and (_lead or _opp_done))"""
assert old in src
new = """                _ok = (_win_lo < d[i] < _win_hi
                       and not self._done[i]
                       and (not _lead or _front_done)
                       and (_lead or _opp_done))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v949b")