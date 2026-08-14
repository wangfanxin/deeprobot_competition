#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                # v994: 悬空轮"收腿"——轮已抬到目标高度但未过棱(fn=0)时，
                # 腿把轮前推(relx 0.15+)的反作用把 body 往后拉，后轮推力被
                # 抵消、狗死锁 8s(实测 fn=0 + 后轮 6-13.5Nm 前驱推不动)。
                # 目标 relx 收到 0.06(轮收在髋下)，后轮推着 body 前进，悬空
                # 轮自然漂过棱后落到台面。
                if float(wheel_xyz[leg, 2]) > float(terrain_h[leg]) + self.fk.r + 0.005:
                    _rel[0] = min(float(_rel[0]), float(os.environ.get(
                        "S10_QP_TUCK_RELX", "0.06")))"""
new = """                # v994/v995: 悬空轮"收腿"——轮高于目标(wheel_z > wz_t+0.01)
                # 即悬空。悬空时腿把轮前推(relx 0.15+)的反作用把 body 往后
                # 拉，后轮推力被抵消、狗死锁 8s(实测 fn=0 + 后轮 6-13.5Nm
                # 前驱推不动)。目标 relx 收到 TUCK_RELX(0.06)，后轮推着
                # body 前进，悬空轮自然漂过棱后落到台面。
                if float(wheel_xyz[leg, 2]) > _wz_t + 0.01:
                    _rel[0] = min(float(_rel[0]), float(os.environ.get(
                        "S10_QP_TUCK_RELX", "0.06")))"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched")