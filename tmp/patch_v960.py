# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """        # λ_ref：支撑 mg/4 均载，抬升 0
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        for i in range(4):
            if stance_mask[i] > 0.5:
                lam_ref[i, 2] = self.m * self.g / 4.0"""
assert old in src
new = """        # λ_ref：支撑 mg/4 均载，抬升 0
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        # v960: 后轮爬顶期支撑腿 λ_ref 提到 mg/3——抬升腿反作用把 body
        # 对侧压起（RR 爬时左轮被抬到 1.17 实测），三支撑腿多承载压住
        # body 抵消反作用
        _rear_sw_ref = float(np.max(step_lift[2:4])) > 0.5
        _lam_st = self.m * self.g / (3.0 if _rear_sw_ref else 4.0)
        for i in range(4):
            if stance_mask[i] > 0.5:
                lam_ref[i, 2] = _lam_st"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v960")