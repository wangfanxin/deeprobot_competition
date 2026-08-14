# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py")
src = p.read_text(encoding="utf-8-sig")
old = """            _rc = float(np.clip((self.kp_roll * (-float(body["roll"]))
                                 - _kdr * roll_rate) * 0.0025,
                                -0.05, 0.05))"""
assert old in src
new = """            _rc_max = float(os.environ.get("S10_FP_ROLL_RC_MAX", "0.10"))
            _rc = float(np.clip((self.kp_roll * (-float(body["roll"]))
                                 - _kdr * roll_rate) * 0.0025,
                                -_rc_max, _rc_max))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")