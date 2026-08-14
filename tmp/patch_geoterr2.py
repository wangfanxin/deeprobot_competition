# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
old = """                        for _i in range(4):
                            _kz = float(wheel_xyz[_i, 2] - 0.081)
                            terr[_i] = min(_kz, float(_geo_ref[_i]) - 0.081)
                    except Exception:
                        pass"""
assert old in src
new = """                        for _i in range(4):
                            _kz = float(wheel_xyz[_i, 2] - 0.081)
                            _gz = float(_geo_ref[_i]) - 0.081
                            # 几何参考合理范围保护（stair_wheel_ref 表未覆盖
                            # 时返回 0 → min 拉到负值 → ground_f=0 轮矩全灭
                            # 实测）
                            if 0.4 < _gz < 1.5:
                                terr[_i] = min(_kz, _gz)
                    except Exception:
                        pass"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")