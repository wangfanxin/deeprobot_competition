#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        cmd = dict(cmd)
        cmd["step_lift"] = step_lift
        cmd["place_z"] = place_z
        # 终版：支撑腿用几何地形（世界坐标），不用 lidar
        terrain_h = self._geo_terrain(wheel_xyz)"""
new = """        cmd = dict(cmd)
        cmd["step_lift"] = step_lift
        cmd["place_z"] = place_z
        # 终版：支撑腿用几何地形（世界坐标），不用 lidar
        terrain_h = self._geo_terrain(wheel_xyz)
        # 后轴 SWING 期前腿加深静压（抗后轴抬升反作用，终版"后腿主动加
        # 垂直力抗抬头"对应项）——后轴抬轮把 body 后部顶起，前腿压载
        # 把 body 拉平，防前轮被反作用折叠上翻（0.86-0.91 实测）。
        _press_base = float(os.environ.get("S10_FP_PRESS", "0.005"))
        if self._sp_r > 0.5:
            os.environ["S10_FP_PRESS"] = str(float(os.environ.get(
                "S10_FP_PRESS_REAR", "0.030")))
        elif float(os.environ.get("S10_FP_PRESS", "0.005")) != _press_base:
            os.environ["S10_FP_PRESS"] = str(_press_base)"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")