# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            # v933: roll/pitch 修正只在后轮爬顶期生效（2 点前支撑最需要）
            # ——前轮爬升期启用会通过 2 后腿产生过伸/λ 爆炸（实测 0.99
            # 过伸翻车）；后轮爬顶时前腿在台面上，修正力有支撑可作用
            _rear_sw = float(np.max(step_lift[2:4])) > 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0")) if _rear_sw else 0.0
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0")) if _rear_sw else 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]"""
assert old in src
new = """            # v933b: roll/pitch 修正只在后轮爬顶期生效（2 点前支撑最需要）
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
src = src.replace(old, new, 1)

# 在 compute_tau 存 _rear_swing（调用 _qp_solve 前）
old2 = """        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)"""
assert old2 in src
new2 = """        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)
        self._rear_swing = float(np.max(step_lift[2:4])) > 0.5"""
src = src.replace(old2, new2, 1)
p.write_text(src, encoding="utf-8")
print("patched v933b")