# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py")
src = p.read_text(encoding="utf-8-sig")
old = """                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    if _dhv <= 0.085:
                        continue
                    _dd = float(np.dot(_ax_xy - _rp, _tng))
                    if -0.20 < _dd < 0.05 and abs(_dd) < abs(_best_d):
                        _best_d = _dd; _best = (_rp, _tng, _dhv, _top)"""
assert old in src
new = """                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    if _dhv <= 0.085:
                        continue
                    _dd = float(np.dot(_ax_xy - _rp, _tng))
                    # v921: 搜索窗放宽到 SWING 触发点(_cl)——原 [-0.20,0.05]
                    # 比 SWING 窗(0.30)窄，d∈(0.05,0.30] 用主循环 place_z
                    # (=0.666) 目标抬到 0.787 过伸实测；v920 贴面轮廓在
                    # d>R+0.02 时给平地 flat+r，不会提前抬
                    if -_cl < _dd < 0.05 and abs(_dd) < abs(_best_d):
                        _best_d = _dd; _best = (_rp, _tng, _dhv, _top)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched stw search window")