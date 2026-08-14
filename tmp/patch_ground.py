# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
old = """            stair_world = [(_pt2, _tng2, _arc2, 0.125, 0.666)]
            stair_risers = [(_arc2, 0.125)]
            print('[VMC] BENCH 手工 riser2 单级: y=%.2f arc=%.2f'
                  % (float(_pt2[1]), _arc2), flush=True)"""
assert old in src
new = """            stair_world = [(_pt2, _tng2, _arc2, 0.125, 0.666)]
            stair_risers = [(_arc2, 0.125)]
            # v918: 台架平台高 0.54（接近 box 顶面）——几何表 STAIR_GROUND
            # 默认 0.48 会让 stair_wheel_ref 低估平地 → 后轮悬空锁死实测
            try:
                fol.STAIR_GROUND = 0.54
            except Exception:
                pass
            print('[VMC] BENCH 手工 riser2 单级: y=%.2f arc=%.2f'
                  % (float(_pt2[1]), _arc2), flush=True)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")