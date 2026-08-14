# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """        for _leg in range(4):
            if step_lift[_leg] <= 0.02:
                continue
            _ax_idx = (0, 1) if _leg in (0, 1) else (2, 3)
            _ax_xy = np.mean([wheel_xyz[_i, :2] for _i in _ax_idx], axis=0)"""
assert old in src
new = """        for _leg in range(4):
            if step_lift[_leg] <= 0.02:
                continue
            # v949: 单轮序列——贴面目标用该轮自身位置（不再用轴均值，
            # 否则左右轮同相抬升破坏序列）
            _ax_xy = wheel_xyz[_leg, :2]"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched face per-wheel")