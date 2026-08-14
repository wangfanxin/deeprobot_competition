#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

# add geometric terrain helper before compute_tau
old = """    # ---------------- 主入口 ----------------
    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):"""
new = """    # ---------------- 几何地形（终版：世界坐标，不用 lidar） ----------------
    def _geo_terrain(self, wheel_xyz):
        \"\"\"每轮支撑面高（世界坐标几何）：已过 riser → 最高已过顶；未过 →
        最近 riser 底（当前平台/地面）。替代 lidar terrain_h——棱口 lidar
        读高 0.7+ 会把支撑腿目标泵到 0.78+、body 抬到 1.0（首跑实测）。\"\"\"
        terr = []
        for leg in range(4):
            gt = 0.0
            best_d = 1e9
            best_bot = None
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                _dd = float(np.dot(wheel_xyz[leg, :2] - _rp, _tng))
                if _dd > 0.0:
                    gt = max(gt, float(_top))
                if abs(_dd) < abs(best_d):
                    best_d = _dd
                    best_bot = float(_top - _dhv)
            if gt > 0.4:
                terr.append(gt)
            elif best_bot is not None:
                terr.append(best_bot)
            else:
                terr.append(float(terrain_h[leg]))
        return np.asarray(terr, dtype=np.float64)

    # ---------------- 主入口 ----------------
    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):"""
assert old in src, "edit1"
src = src.replace(old, new)

# use geo terrain in compute_tau
old = """        cmd = dict(cmd)
        cmd["step_lift"] = step_lift
        cmd["place_z"] = place_z"""
new = """        cmd = dict(cmd)
        cmd["step_lift"] = step_lift
        cmd["place_z"] = place_z
        # 终版：支撑腿用几何地形（世界坐标），不用 lidar
        terrain_h = self._geo_terrain(wheel_xyz)"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")