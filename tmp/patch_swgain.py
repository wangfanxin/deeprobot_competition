# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py")
src = p.read_text(encoding="utf-8-sig")
old = """            if _posmode_fp > 0:
                _kpp9 = float(os.environ.get('S10_FP_KP_POS', '0'))
                _kp_leg = (float(_kpp9) if _kpp9 > 0 else self.kp)
                _kd_leg = self.kd"""
assert old in src
new = """            if _posmode_fp > 0:
                _kpp9 = float(os.environ.get('S10_FP_KP_POS', '0'))
                _kp_leg = (float(_kpp9) if _kpp9 > 0 else self.kp)
                _kd_leg = self.kd
                # v922: posmode 抬升腿单独软增益（默认 40，支撑 120）——
                # 全增益抬升腿把 body 顶高(1.11 实测)而非抬轮（轮贴地推不
                # 动，反力顶 body）；软增益让轮顺立面滚上、腿只引导不过推
                if sl > 0.1:
                    _kpsw = float(os.environ.get('S10_FP_KP_SW', '40'))
                    _kp_leg = _kpsw
                    _kd_leg = float(os.environ.get('S10_FP_KD_SW', '3'))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")