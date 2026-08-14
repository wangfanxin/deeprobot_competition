# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py")
src = p.read_text(encoding="utf-8-sig")
old = """            if sl <= 0.5:
                wz = min(wz, float(terrain_h[leg]) + self.fk.r
                         - _fp_press + 0.05)"""
assert old in src
new = """            if sl <= 0.5:
                wz = min(wz, float(terrain_h[leg]) + self.fk.r
                         - _fp_press + 0.05)
                # v919b: 几何台面顶封顶——轮在台面上时运动学地面锁死过伸
                # （前轮上台后悬空 0.15m 卡在台上不进实测）；拉回
                # geo_top+r+0.01 让轮落回台面抓地
                try:
                    _gt = float(cmd.get("geo_top", np.zeros(4))[leg])
                    if _gt > 0.4:
                        wz = min(wz, _gt + self.fk.r + 0.01)
                except Exception:
                    pass"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched fp")