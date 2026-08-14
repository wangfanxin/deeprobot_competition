#!/usr/bin/env python3
import io

# 1) base: add leg kwarg to _ik signature + call site
p1 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_vmc_legs.py"
s1 = io.open(p1, "r", encoding="utf-8").read()
old = """    def _ik(self, xd, zd, q1, q2, lift=False):"""
new = """    def _ik(self, xd, zd, q1, q2, lift=False, leg=None):"""
assert old in s1, "sig anchor"
s1 = s1.replace(old, new)
old = """            q1t, q2t = self._ik(float(rel[0]), _rz, q1, q2, lift=(sl > 0.1))"""
new = """            q1t, q2t = self._ik(float(rel[0]), _rz, q1, q2,
                                 lift=(sl > 0.1), leg=leg)"""
assert old in s1, "call anchor"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)

# 2) StairWBC: fix branch by leg (front +acos, rear -acos)
p2 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
s2 = io.open(p2, "r", encoding="utf-8").read()
old = """    def _ik(self, xd, zd, q1, q2, lift=False):
        \"\"\"闭式解（q2=+acos 自然膝分支），一次到位、不收敛到镜像折叠解。
        基类 lift=True 强制 q2>=1.8 → 轮折叠到髋上方（贴面爬升过伸 1.1+
        实测）；贴面爬升轮应保持髋下（q2 自由）。lift 参数忽略。\"\"\"
        L1, L2 = self.fk.L1, self.fk.L2
        zd_d = -zd   # 基类 zd 向上为正 -> 闭式用向下为正
        r2 = min(xd * xd + zd_d * zd_d, (L1 + L2) ** 2 - 1e-6)
        c2 = float(np.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2),
                           -1.0, 1.0))
        q2n = float(np.arccos(c2))
        # 分支按当前 q2 符号：前腿标称 q2>0 用 +acos，后腿标称 q2<0 用
        # -acos（镜像腿）。固定 +acos 会让后腿折叠上翻 1.35+（实测）。
        if q2 < 0:
            q2n = -q2n
        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        q1n = float(np.clip(q1n, -1.7, 1.0))
        q2n = float(np.clip(q2n, -1.0, 3.0))
        return q1n, q2n"""
new = """    def _ik(self, xd, zd, q1, q2, lift=False, leg=None):
        \"\"\"闭式解，一次到位、不收敛到镜像折叠解。分支按腿固定：
        前腿(0,1) q2=+acos，后腿(2,3) q2=-acos（镜像腿）——按当前 q2
        符号选分支会卡在镜像折叠(一旦折叠 q2 变号→一直选错分支→永远
        折叠，FR 0.86/RR 0.81 实测)。lift 参数忽略（贴面爬升轮保持髋下）。\"\"\"
        L1, L2 = self.fk.L1, self.fk.L2
        zd_d = -zd   # 基类 zd 向上为正 -> 闭式用向下为正
        r2 = min(xd * xd + zd_d * zd_d, (L1 + L2) ** 2 - 1e-6)
        c2 = float(np.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2),
                           -1.0, 1.0))
        q2n = float(np.arccos(c2))
        if leg is not None and leg in (2, 3):
            q2n = -q2n
        elif leg is None and q2 < 0:
            q2n = -q2n
        q1n = float(np.arctan2(xd, zd_d) - np.arctan2(
            L2 * np.sin(q2n), L1 + L2 * np.cos(q2n)))
        q1n = float(np.clip(q1n, -1.7, 1.0))
        q2n = float(np.clip(q2n, -1.0, 3.0))
        return q1n, q2n"""
assert old in s2, "stw anchor"
s2 = s2.replace(old, new)
io.open(p2, "w", encoding="utf-8").write(s2)
print("patched OK")