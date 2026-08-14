# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
old = """        else:
            terr = np.array([terrain_at(wheel_xyz[i, 0], wheel_xyz[i, 1])
                             for i in range(4)])
        # v223b: 地形低通（lidar 栅格稀疏/噪声，防腿抖）"""
assert old in src
new = """        else:
            terr = np.array([terrain_at(wheel_xyz[i, 0], wheel_xyz[i, 1])
                             for i in range(4)])
            # v908: KIN fallback 在 LOOKAHEAD=0 时也生效（台架出生 lidar
            # 地图为空→后轮格 0.0→ground_f=0 轮矩全灭实测）
            if (os.environ.get('S10_VMC_TERRAIN_KIN', '0') == '1'
                    and os.environ.get('S10_VMC_TERRAIN', 'ray') == 'lidar'):
                for _i in range(4):
                    if not lterr.has(wheel_xyz[_i, 0], wheel_xyz[_i, 1]):
                        terr[_i] = float(wheel_xyz[_i, 2] - 0.081)
        # v223b: 地形低通（lidar 栅格稀疏/噪声，防腿抖）"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")