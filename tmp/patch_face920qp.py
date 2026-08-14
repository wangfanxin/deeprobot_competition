# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            # v911: 贴面 ramp 覆盖棱前窗 [0, cl]——原版 d>0 直接给台面顶
            # 目标，一进 SWING 窗就猛抬轮 0.13m（台架 front wz 0.62→1.0
            # 过冲、pitch/roll 级联实测）。改为棱前 cl 内从地面平滑 ramp
            # 到台面顶（d=cl 时=地面+r，d=0 时=台面顶+r），棱后保持台面顶。
            if 0.0 <= _d_w <= _cl:
                _t = float(np.clip(1.0 - _d_w / max(_cl, 1e-6), 0.0, 1.0))
                _ss = _t * _t * (3.0 - 2.0 * _t)
                _z_face = min(_z_bot + _r + _dhv * _ss, _top + _r + 0.005)
            else:
                _z_face = min(_top + _r, _top + _r + 0.005)
            pz[_leg] = _z_face - _r - _margin
        return pz"""
assert old in src
new = """            # v920: 贴面轮廓修正（同 StairWBC）——d>=R+0.02 平地 flat+r；
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
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched qp face v920")