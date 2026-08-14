# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
old = """                terr = np.array(
                    [terrain_at(wheel_xyz[i, 0], wheel_xyz[i, 1])
                     for i in range(4)], dtype=np.float64)
            except Exception:
                pass"""
assert old in src
new = """                terr = np.array(
                    [terrain_at(wheel_xyz[i, 0], wheel_xyz[i, 1])
                     for i in range(4)], dtype=np.float64)
                # v908: stair 覆盖分支同样启用 KIN fallback（台架出生 lidar
                # 空图 → 后轮格 0.0 → ground_f=0 轮矩全灭实测）
                if (os.environ.get('S10_VMC_TERRAIN_KIN', '0') == '1'
                        and os.environ.get('S10_VMC_TERRAIN', 'ray') == 'lidar'):
                    for _i in range(4):
                        if not lterr.has(wheel_xyz[_i, 0], wheel_xyz[_i, 1]):
                            terr[_i] = float(wheel_xyz[_i, 2] - 0.081)
            except Exception:
                pass"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")