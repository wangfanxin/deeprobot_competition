# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            # v920: 贴面轮廓修正（同 StairWBC）——d>=R+0.02 平地 flat+r；
            # [R+0.02, R-0.06] 平滑抬到 top+r；再后台面顶+r。原 [0,cl]
            # ramp 提前抬轮悬空失接触实测。
            _lift_lo = _r - 0.06
            _lift_hi = _r + 0.02
            if _d_w >= _lift_hi:
                _z_face = _z_bot + _r
            elif _d_w >= _lift_lo:
                _t = float(np.clip(
                    (_lift_hi - _d_w) / max(_lift_hi - _lift_lo, 1e-6),
                    0.0, 1.0))
                _ss = _t * _t * (3.0 - 2.0 * _t)
                _z_face = min(_z_bot + _r + _dhv * _ss, _top + _r + 0.005)
            else:
                _z_face = min(_top + _r, _top + _r + 0.005)
            pz[_leg] = _z_face - _r - _margin
        return pz"""
assert old in src
new = """            # v920/v924: 贴面轮廓（同 StairWBC）——抬升窗 [0.15,-0.05]
            _lift_hi = float(os.environ.get("S10_STAIR_LIFT_HI", "0.15"))
            _lift_lo = -0.05
            if _d_w >= _lift_hi:
                _z_face = _z_bot + _r
            elif _d_w >= _lift_lo:
                _t = float(np.clip(
                    (_lift_hi - _d_w) / max(_lift_hi - _lift_lo, 1e-6),
                    0.0, 1.0))
                _ss = _t * _t * (3.0 - 2.0 * _t)
                _z_face = min(_z_bot + _r + _dhv * _ss, _top + _r + 0.005)
            else:
                _z_face = min(_top + _r, _top + _r + 0.005)
            pz[_leg] = _z_face - _r - _margin
        return pz"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched qp")