# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py")
src = p.read_text(encoding="utf-8-sig")
old = """                # v914: 真实贴面弧线（用户 A+B 公式）——
                #   d >= R:      轮贴地 z = 底+r（不提前悬空）
                #   R > d >= 0:  沿立面滚 z = 底 + sqrt(R^2 - d^2)
                #   -0.06<=d<0:  过棱抬升 z: 底+R -> 顶+r（清台面 h-R 余量）
                #   d < -0.06:   台面顶+r
                if _d_w >= _r:
                    _z_face = _z_bot + _r
                elif _d_w >= 0.0:
                    _z_face = _z_bot + float(np.sqrt(max(_r * _r - _d_w * _d_w, 0.0)))
                elif _d_w >= -0.06:
                    _t = float(np.clip((-_d_w) / 0.06, 0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    _z_face = _z_bot + _r + (_dhv - _r) * _ss
                else:
                    _z_face = _top + _r"""
assert old in src
new = """                # v920: 贴面轮廓修正（FP 调试实测）——sqrt 弧线在 d∈[0,R]
                # 目标低于平地轮高(0.541 vs 0.626)，腿目标比轮低 → 腿把
                # body 顶高 0.2m、IK 分支乱跳（q1t 0.9↔-0.01 实测）。
                # 正确轮廓：d>=R+0.02 平地 flat+r；[R+0.02, R-0.06] 平滑抬
                # 到 top+r（棱口处抬升清台面）；再后保持台面顶+r。
                _lift_lo = _r - 0.06
                _lift_hi = _r + 0.02
                if _d_w >= _lift_hi:
                    _z_face = _z_bot + _r
                elif _d_w >= _lift_lo:
                    _t = float(np.clip(
                        (_lift_hi - _d_w) / max(_lift_hi - _lift_lo, 1e-6),
                        0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    _z_face = _z_bot + _r + _dhv * _ss
                else:
                    _z_face = _top + _r"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched stw face v920")