#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# swing anti-over: normalized gradient
old = """                # v985/v986: 防过伸力矩经 Jacobian 投影到 hipy+knee——轮折
                # 叠到 q1+q2≈π 时膝盖对轮高近奇异(无权威)，只打膝盖推不动
                # 轮。J^T [0, +F]（+z 为向下）让大腿也参与压轮。
                # v986 修正: 原写 [0,-F] 是向上推(符号反了)。
                if abs(self._ov_sw_f[leg]) > 0.5:
                    _tov = J.T @ np.array([0.0, self._ov_sw_f[leg]])
                    tau[hipy_i] += float(_tov[0])
                    tau[knee_i] += float(_tov[1])"""
new = """                # v987: 防过伸沿 pz 增大梯度方向施加满力矩——J^T 投影在折叠
                # 位姿(q1+q2≈π)下近奇异，45N 力只剩 ~5Nm 关节力矩推不动轮
                # (v985/986 实测轮仍悬空 5cm+)。归一化梯度任何姿态都满幅。
                if abs(self._ov_sw_f[leg]) > 0.5:
                    _Jz = np.array([J[1, 0], J[1, 1]])
                    _nz = float(np.linalg.norm(_Jz)) + 1e-6
                    _ovg = float(np.clip(self._ov_sw_f[leg], -48.0, 48.0))
                    _tov = _Jz / _nz * _ovg
                    tau[hipy_i] += float(_tov[0])
                    tau[knee_i] += float(_tov[1])"""
assert old in src, "edit1"
src = src.replace(old, new)

# stance anti-over: normalized gradient
old = """                    # v985/v986: 经 Jacobian 投影(同 swing)——折叠位姿下大腿
                    # 参与压轮；[0,+F] 为向下(+z-down)
                    _tov2 = J.T @ np.array([0.0, self._ov_st_f[leg]])
                    th += float(_tov2[0])
                    tk += float(_tov2[1])"""
new = """                    # v987: 经 pz 梯度方向(同 swing)——折叠位姿下满幅压轮
                    _Jz2 = np.array([J[1, 0], J[1, 1]])
                    _nz2 = float(np.linalg.norm(_Jz2)) + 1e-6
                    _ovg2 = float(np.clip(self._ov_st_f[leg], -48.0, 48.0))
                    _tov2 = _Jz2 / _nz2 * _ovg2
                    th += float(_tov2[0])
                    tk += float(_tov2[1])"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")