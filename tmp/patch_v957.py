# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    if _over_s > 0.01:
                        _k_ovs = float(os.environ.get("S10_QP_K_OVER", "150.0"))
                        tk -= _k_ovs * _over_s"""
assert old in src
new = """                if _gt_hi > 0.4:
                    _over_s = float(wheel_xyz[leg, 2]) - (_gt_hi + self.fk.r)
                    if _over_s > 0.01:
                        # v957: 支撑腿防过伸用更高增益(默认1000)——前轮悬空
                        # 0.03-0.06 无抓地、狗不前进、RR 进不了窗实测
                        _k_ovs = float(os.environ.get("S10_QP_K_OVER_ST", "1000.0"))
                        tk -= _k_ovs * _over_s"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v957")