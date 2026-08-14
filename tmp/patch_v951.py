# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            # v933b: roll/pitch 修正只在后轮爬顶期生效（2 点前支撑最需要）
            # ——前轮爬升期启用会通过 2 后腿产生过伸/λ 爆炸（实测 0.99
            # 过伸翻车）；后轮爬顶时前腿在台面上，修正力有支撑可作用。
            # v933 引用了 _qp_solve 里不存在的 step_lift → NameError 被吞
            # → QP 又静默回退均载（v939 台架"稳定8s"实为位置基行为实测）；
            # 改用 compute_tau 存入的 self._rear_swing。
            _rear_sw = float(getattr(self, '_rear_swing', 0.0)) > 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0")) if _rear_sw else 0.0
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0")) if _rear_sw else 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]"""
assert old in src
new = """            # v951: roll/pitch 修正在单轮序列下全 swing 期启用——v933 为
            # 2 点轴 swing 关闭（载荷不对称崩）；单轮序列恒 3 点支撑，
            # roll 修正安全且必要（FL 爬时 body 侧滚 15° 致 FR 髋低过折叠
            # 过伸实测）。yaw 阻尼仍按 v947 只后轮爬顶期。
            _any_sw2 = float(getattr(self, '_any_swing', 0.0)) > 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0")) if _any_sw2 else 0.0
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0")) if _any_sw2 else 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v951")