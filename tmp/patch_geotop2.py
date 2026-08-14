# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
# 2) 主循环算每轮几何台面顶，传给 FP（插入到 climb_mask 前）
old2 = """            # v813: 计算每轮爬升窗掩码（轮世界 y 与 riser y 距离）
            _climb_mask = np.zeros(4)"""
assert old2 in src, "climb anchor not found"
new2 = """            # v919: 每轮几何台面顶（支撑腿过伸封顶用）——取该轮 y 之前最近
            # 的 riser 台面（轮在台面上=台面顶；平地=STAIR_GROUND）。FP 支撑
            # 腿目标封顶 geo_top+r，防运动学地面锁死过伸（前轮上台后悬空
            # 0.15m 卡在台上不进实测）
            _geo_top4 = np.full(4, float(getattr(fol, 'STAIR_GROUND', 0.48)))
            try:
                if _in_stairzone_now and stair_world:
                    for (_rp, _tng, _sr, _dhv, _top) in stair_world:
                        for _i in range(4):
                            _di = float(np.dot(
                                wheel_xyz[_i, :2] - _rp, _tng))
                            if _di <= 0.0 and float(_top) > _geo_top4[_i]:
                                _geo_top4[_i] = float(_top)
            except Exception:
                pass
            # v813: 计算每轮爬升窗掩码（轮世界 y 与 riser y 距离）
            _climb_mask = np.zeros(4)"""
src = src.replace(old2, new2, 1)
p.write_text(src, encoding="utf-8")
print("patched geotop")