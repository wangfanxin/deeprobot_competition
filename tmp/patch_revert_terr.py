# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
# 1) 地形回退为纯运动学（v916 原版）
old = """                # v916/v917: 楼梯执行区支撑腿地面 = min(运动学地面, 几何台面)——
                # 运动学地面防 lidar 污染（后轮格读到 riser 顶 → 后腿猛伸离地
                # 失稳实测）；几何台面(stair_wheel_ref)防运动学锁死过伸（前轮
                # 上台面后腿过伸悬空 0.15m → 拉回台面顶，狗卡在台上不进实测）。
                # 平地：运动学=几何=平地；棱前 ramp：min=运动学(不提前抬)；
                # 台面：min=几何顶(0.666)。S10_STAIR_KIN_TERR 可关。
                if float(os.environ.get('S10_STAIR_KIN_TERR', '1')) > 0:
                    try:
                        _geo_ref = np.asarray(fol.stair_wheel_ref(
                            np.asarray([wheel_xyz[_i, 1]
                                        for _i in range(4)])), dtype=np.float64)
                        for _i in range(4):
                            _kz = float(wheel_xyz[_i, 2] - 0.081)
                            _gz = float(_geo_ref[_i]) - 0.081
                            # v918b: 纯 min——v918 已把 bench STAIR_GROUND
                            # 修正为 0.54（接近 box 顶），几何平地正确后：
                            # 平地 min=运动学；过伸轮 min=几何顶(拉回台面)；
                            # 悬空后轮 min=平地(落回)。范围保护防表未覆盖。
                            if 0.4 < _gz < 1.5:
                                terr[_i] = min(_kz, _gz)
                    except Exception:
                        pass
            except Exception:
                pass"""
assert old in src, "old terrain block not found"
new = """                # v916: 楼梯执行区支撑腿地面 = 运动学地面（轮心 z - r）——
                # lidar 高程在楼梯区失真（轮下格被 riser 顶污染 → 后腿猛伸
                # 离地失稳实测）。S10_STAIR_KIN_TERR 可关。
                if float(os.environ.get('S10_STAIR_KIN_TERR', '1')) > 0:
                    for _i in range(4):
                        terr[_i] = float(wheel_xyz[_i, 2] - 0.081)
            except Exception:
                pass"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched terrain back to kinematic")