# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            if _d_w <= 0.0:
                if _d_w >= -_cl:
                    _t = float(np.clip((_d_w + _cl) / max(_cl, 1e-6), 0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    _z_face = min(_z_bot + _r + _dhv * _ss, _top + _r + 0.005)
                else:
                    _z_face = _z_bot + _r
            else:
                _z_face = min(_top + _r, _top + _r + 0.005)"""
assert old in src
new = """            # v939: 抬升窗收紧到 [-0.08, 0]（轮半径 0.081，d=-0.08 时轮
            # 正好贴棱）——v901 的 [-cl,0] 窗在轮还在地上 0.3m 时就开始抬
            # → 轮推不动反顶 body（0.96 泵高、后腿够不到、roll 崩实测）。
            # 棱口才抬，动量+贴面把轮带上去。
            if _d_w <= 0.0:
                if _d_w >= -0.08:
                    _t = float(np.clip((_d_w + 0.08) / 0.08, 0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    _z_face = min(_z_bot + _r + _dhv * _ss, _top + _r + 0.005)
                else:
                    _z_face = _z_bot + _r
            else:
                _z_face = min(_top + _r, _top + _r + 0.005)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v939")