# -*- coding: utf-8 -*-
import io
files = {
 "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py": [
   ("            if _dhv <= 0.085:      # 小台阶纯滚（首级 0.063m < 轮半径）",
    "            if _dhv <= 0.050:      # v1027: riser1(0.061m) 也走 SWING 抬轮放置（锐角滚动对偏航极敏感）"),
   ("            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:\n                if _dhv <= 0.085:\n                    continue\n                _dd = float(np.dot(_ax_xy - _rp, _tng))",
    "            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:\n                if _dhv <= 0.050:\n                    continue\n                _dd = float(np.dot(_ax_xy - _rp, _tng))"),
 ],
 "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py": [
   ("                for (_rp, _tng, _sr, _dhv, _top) in stair_world:\n                    if _dhv <= 0.085:\n                        continue\n                    _ddc = float(np.dot(_fax_c - _rp, _tng))",
    "                for (_rp, _tng, _sr, _dhv, _top) in stair_world:\n                    if _dhv <= 0.050:\n                        continue\n                    _ddc = float(np.dot(_fax_c - _rp, _tng))"),
   ("                for (_rp0, _tng0, _sr0, _dh0, _top0) in stair_world:\n                    if _dh0 <= 0.085:\n                        continue\n                    _dd0 = float(np.dot(body_pos[:2] - _rp0, _tng0))",
    "                for (_rp0, _tng0, _sr0, _dh0, _top0) in stair_world:\n                    if _dh0 <= 0.050:\n                        continue\n                    _dd0 = float(np.dot(body_pos[:2] - _rp0, _tng0))"),
   ("                for (_rp, _tng, _sr, _dhv, _top) in stair_world:\n                # 只对高 riser（>轮半径 0.085）触发抬升，小台阶纯滚\n                if _dhv <= 0.085:\n                    continue\n                _dd_f = float(np.dot(_fax_p - _rp, _tng))",
    "                for (_rp, _tng, _sr, _dhv, _top) in stair_world:\n                # v1027: riser1 也触发抬升（锐角滚动对偏航敏感）\n                if _dhv <= 0.050:\n                    continue\n                _dd_f = float(np.dot(_fax_p - _rp, _tng))"),
 ],
}
for p, subs in files.items():
    s = io.open(p, encoding="utf-8").read()
    for old, new in subs:
        assert old in s, (p, old[:60])
        s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched", p)