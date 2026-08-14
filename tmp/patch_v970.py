#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """            _rear_sw = float(getattr(self, '_rear_swing', 0.0)) > 0.5
            # v969: roll/pitch 修正任意 SWING 期启用(原仅后轮爬顶)——前轮
            # 单轮抬升时 body 无俯仰控制 → 后仰 0.8-1.0rad 实测；现在有接触
            # 感知支撑+重心解 λ_ref+抬升反作用力矩，QP 有可行域做修正
            _any_sw_q = float(np.min(stance_mask)) < 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0"))
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0"))
            if not _any_sw_q:
                _ar_k = 0.0; _ap_k = 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]"""
new = """            _rear_sw = float(getattr(self, '_rear_swing', 0.0)) > 0.5
            # v970: roll/pitch 修正按 SWING 模式(step_lift)激活，不是接触
            # 掩码——贴面期 SWING 轮还在地上(qp_stance=1)，用接触掩码会
            # 把姿态控制全程关掉(v969 无效改动实测)。前轮单轮抬升期 body
            # 无俯仰控制 → 后仰 0.8-1.0rad；现在有接触感知+重心解+反作用
            # 力矩，QP 有可行域做修正。
            _any_sw_q = float(np.max(getattr(self, '_step_lift_last',
                                             np.zeros(4)))) > 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0"))
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0"))
            if not _any_sw_q:
                _ar_k = 0.0; _ap_k = 0.0
            a_des[3] = _ar_k * body["roll"]
            a_des[4] = _ap_k * body["pitch"]"""
assert old in src, "edit1 anchor missing"
src = src.replace(old, new)

# store step_lift for _qp_solve
old = """        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)"""
new = """        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)
        self._step_lift_last = step_lift.copy()"""
assert old in src, "edit2 anchor missing"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")