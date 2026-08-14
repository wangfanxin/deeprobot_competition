#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """    # ---------------- 几何地形（终版：世界坐标，不用 lidar） ----------------"""
new = """    # ---------------- 闭式 2 连杆 IK（覆盖基类迭代式） ----------------
    def _ik(self, xd, zd, q1, q2, lift=False):
        \"\"\"闭式解（q2=+acos 自然膝分支），一次到位、不收敛到镜像折叠解。
        基类 lift=True 强制 q2>=1.8 → 轮折叠到髋上方（贴面爬升过伸 1.1+
        实测）；贴面爬升轮应保持髋下（q2 自由）。lift 参数忽略。\"\"\"
        L1, L2 = self.fk.L1, self.fk.L2
        r2 = min(xd * xd + zd * zd, (L1 + L2) ** 2 - 1e-6)
        c2 = float(np.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2),
                           -1.0, 1.0))
        q2n = float(np.arccos(c2))
        q1n = float(np.arctan2(xd, zd) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        q1n = float(np.clip(q1n, -1.7, 1.0))
        q2n = float(np.clip(q2n, -1.0, 3.0))
        return q1n, q2n

    # ---------------- 几何地形（终版：世界坐标，不用 lidar） ----------------"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")