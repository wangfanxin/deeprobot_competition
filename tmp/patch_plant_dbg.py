#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                    # v987: 经 pz 梯度方向(同 swing)——折叠位姿下满幅压轮
                    _Jz2 = np.array([J[1, 0], J[1, 1]])
                    _nz2 = float(np.linalg.norm(_Jz2)) + 1e-6
                    _ovg2 = float(np.clip(self._ov_st_f[leg], -48.0, 48.0))
                    _tov2 = _Jz2 / _nz2 * _ovg2
                    th += float(_tov2[0])
                    tk += float(_tov2[1])"""
new = """                    # v987: 经 pz 梯度方向(同 swing)——折叠位姿下满幅压轮
                    _Jz2 = np.array([J[1, 0], J[1, 1]])
                    _nz2 = float(np.linalg.norm(_Jz2)) + 1e-6
                    _ovg2 = float(np.clip(self._ov_st_f[leg], -48.0, 48.0))
                    _tov2 = _Jz2 / _nz2 * _ovg2
                    th += float(_tov2[0])
                    tk += float(_tov2[1])
                    if float(os.environ.get("S10_QP_DEBUG", "0")) > 2:
                        _hipz3 = float(_hip_w2[2])
                        print('[PLANT] t=%.2f leg=%d q1=%.2f q2=%.2f '
                              'q1t=%.2f q2t=%.2f bz=%.3f hipz=%.3f '
                              'wz=%.3f wzt=%.3f ovf=%.1f th=%.1f tk=%.1f'
                              % (self._t, leg, q1, q2, _q1t, _q2t,
                                 body["pos"][2], _hipz3, wheel_xyz[leg, 2],
                                 _wzt, self._ov_st_f[leg], th, tk),
                              flush=True)"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched")