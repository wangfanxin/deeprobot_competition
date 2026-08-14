# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py")
src = p.read_text(encoding="utf-8-sig")
old = """                # v920: 贴面轮廓修正（FP 调试实测）——sqrt 弧线在 d∈[0,R]
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
assert old in src
new = """                # v920/v924: 贴面轮廓——抬升窗放宽到 [0.15, -0.05]（0.2m，
                # 1.2m/s 下 0.17s，0.74m/s 垂直速度适中）。v920 的窄窗
                # [R+0.02, R-0.06] 太紧（67ms 抬 0.125m），软增益抬不动轮、
                # 硬增益泵 body（FP 调试实测）；提前轻悬空、过棱落台面。
                _lift_hi = float(os.environ.get("S10_STAIR_LIFT_HI", "0.15"))
                _lift_lo = -0.05
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
print("patched stw")