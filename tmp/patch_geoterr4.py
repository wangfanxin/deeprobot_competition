# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
old = """                        for _i in range(4):
                            _kz = float(wheel_xyz[_i, 2] - 0.081)
                            _gz = float(_geo_ref[_i]) - 0.081
                            # 只在运动学明显高于几何（过伸>0.08m）时用几何
                            # 封顶——几何 STAIR_GROUND 可能与实际地面差
                            # 0.06m（台架 box 0.54 vs 表 0.479），纯 min 会
                            # 把地形拉低 → ground_f=0 轮矩全灭实测
                            if 0.4 < _gz < 1.5 and _kz > _gz + 0.08:
                                terr[_i] = _gz
                    except Exception:
                        pass"""
assert old in src
new = """                        for _i in range(4):
                            _kz = float(wheel_xyz[_i, 2] - 0.081)
                            _gz = float(_geo_ref[_i]) - 0.081
                            # v918b: 纯 min——v918 已把 bench STAIR_GROUND
                            # 修正为 0.54（接近 box 顶），几何平地正确后：
                            # 平地 min=运动学；过伸轮 min=几何顶(拉回台面)；
                            # 悬空后轮 min=平地(落回)。范围保护防表未覆盖。
                            if 0.4 < _gz < 1.5:
                                terr[_i] = min(_kz, _gz)
                    except Exception:
                        pass"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")