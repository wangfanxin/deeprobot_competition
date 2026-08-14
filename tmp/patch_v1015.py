#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        cmd = dict(cmd)
        cmd["step_lift"] = step_lift
        cmd["place_z"] = place_z"""
new = """        cmd = dict(cmd)
        cmd["step_lift"] = step_lift
        cmd["place_z"] = place_z
        # v1015: 后轴 SWING 期前轮加深静压（显式压台面）——后轴抬升的
        # 反作用把前轮顶起(0.79-1.2 过伸、roll 崩实测)；前轮压 20mm
        # 把 body 拉住、保持前轮接触台面。基类 S10_FP_PRESS 动态设置。
        if self._sp_r > 0.5:
            os.environ["S10_FP_PRESS"] = str(float(os.environ.get(
                "S10_FP_PRESS_REAR_SW", "0.020")))
        elif float(os.environ.get("S10_FP_PRESS", "0.005")) != "0.005":
            os.environ["S10_FP_PRESS"] = "0.005\""""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")