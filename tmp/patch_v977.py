#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# Edit 1: stance IK target - pure geometric when past riser
old = """                if _gt_hi > 0.4:
                    try:
                        # v959: 支撑腿目标压入台面 2mm（原 +0.01 余量让轮
                        # 悬空 0.01-0.06 无抓地、狗不推进实测）
                        _wzt = min(float(terrain_h[leg]) + self.fk.r,
                                   _gt_hi + self.fk.r - 0.002)"""
new = """                if _gt_hi > 0.4:
                    try:
                        # v959: 支撑腿目标压入台面 2mm（原 +0.01 余量让轮
                        # 悬空 0.01-0.06 无抓地、狗不推进实测）
                        # v977: 过棱后纯几何目标——lidar 在棱口读高(0.79)读低
                        # (0.44)都会把目标带偏(0.87/0.52)，轮已在台面上
                        _wzt = float(_gt_hi) + self.fk.r - 0.002"""
assert old in src, "edit1"
src = src.replace(old, new)

# Edit 2: adaptive z_ref support height - pure geometric when past riser
old = """                _sup = np.zeros(4)
                for _i in range(4):
                    _sup[_i] = float(terrain_h[_i]) + self.fk.r
                    _gtz = 0.0
                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        if _dhv <= 0.085:
                            continue
                        _ddz = float(np.dot(wheel_xyz[_i, :2] - _rp, _tng))
                        if _ddz > 0.0:
                            _gtz = max(_gtz, float(_top))
                    if _gtz > 0.4:
                        _sup[_i] = min(_sup[_i], _gtz + self.fk.r)"""
new = """                _sup = np.zeros(4)
                for _i in range(4):
                    _sup[_i] = float(terrain_h[_i]) + self.fk.r
                    _gtz = 0.0
                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        if _dhv <= 0.085:
                            continue
                        _ddz = float(np.dot(wheel_xyz[_i, :2] - _rp, _tng))
                        if _ddz > 0.0:
                            _gtz = max(_gtz, float(_top))
                    if _gtz > 0.4:
                        # v977: 过棱轮用几何台面(不取 min)——lidar 读低
                        # 会把 z_ref 拽下去
                        _sup[_i] = _gtz + self.fk.r"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")