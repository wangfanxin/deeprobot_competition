#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                    _tau_r = _recov_k * max(0.05 - _relx_r, 0.0)
                    # 单位换算：relx 米 → 关节力矩，放大到有效幅值
                    _tau_r = float(np.clip(_tau_r * 20.0, -48.0, 48.0))
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r"""
new = """                    _tau_r = _recov_k * max(0.05 - _relx_r, 0.0)
                    # 终版公式：K_recov=15 Nm/m * relx 误差，温和恢复
                    _tau_r = float(np.clip(_tau_r, -48.0, 48.0))
                    tau[LEG_CTRL_IDX[_leg * 3 + 1]] += _tau_r"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")