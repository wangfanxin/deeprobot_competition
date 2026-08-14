#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                # v985: 防过伸力矩经 Jacobian 投影到 hipy+knee——轮折叠到
                # q1+q2≈π 时膝盖对轮高近奇异(无权威)，只打膝盖推不动轮
                # (悬空 0.9+ 实测)。J^T [0, -F] 让大腿也参与压轮。
                if abs(self._ov_sw_f[leg]) > 0.5:
                    _tov = J.T @ np.array([0.0, -self._ov_sw_f[leg]])
                    tau[hipy_i] += float(_tov[0])
                    tau[knee_i] += float(_tov[1])"""
new = """                # v985/v986: 防过伸力矩经 Jacobian 投影到 hipy+knee——轮折
                # 叠到 q1+q2≈π 时膝盖对轮高近奇异(无权威)，只打膝盖推不动
                # 轮。J^T [0, +F]（+z 为向下）让大腿也参与压轮。
                # v986 修正: 原写 [0,-F] 是向上推(符号反了)。
                if abs(self._ov_sw_f[leg]) > 0.5:
                    _tov = J.T @ np.array([0.0, self._ov_sw_f[leg]])
                    tau[hipy_i] += float(_tov[0])
                    tau[knee_i] += float(_tov[1])"""
assert old in src, "edit1"
src = src.replace(old, new)

old = """                    # v985: 经 Jacobian 投影(同 swing)——折叠位姿下大腿参与
                    th += float((J.T @ np.array([0.0, -self._ov_st_f[leg]]))[0])
                    tk += float((J.T @ np.array([0.0, -self._ov_st_f[leg]]))[1])"""
new = """                    # v985/v986: 经 Jacobian 投影(同 swing)——折叠位姿下大腿
                    # 参与压轮；[0,+F] 为向下(+z-down)
                    _tov2 = J.T @ np.array([0.0, self._ov_st_f[leg]])
                    th += float(_tov2[0])
                    tk += float(_tov2[1])"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")