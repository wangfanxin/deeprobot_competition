# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - 0.78)
            a_des[3] = float(os.environ.get("S10_QP_AR_K", "-20.0")) * body["roll"]
            a_des[4] = float(os.environ.get("S10_QP_AP_K", "-20.0")) * body["pitch"]"""
assert old in src
new = """            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - 0.78)
            # v933: roll/pitch 修正只在后轮爬顶期生效（2 点前支撑最需要）
            # ——前轮爬升期启用会通过 2 后腿产生过伸/λ 爆炸（实测 0.99
            # 过伸翻车）；后轮爬顶时前腿在台面上，修正力有支撑可作用
            _rear_sw = float(np.max(step_lift[2:4])) > 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0")) if _rear_sw else 0.0
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0")) if _rear_sw else 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched qp gated")