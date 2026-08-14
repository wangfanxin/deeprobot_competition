#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# ---------- Edit A: AY front-swing gain smaller ----------
old = """            _any_sw_q = float(np.min(stance_mask)) < 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0"))
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0"))
            _ay_k = float(os.environ.get("S10_QP_AY_K", "-20.0"))"""
new = """            _any_sw_q = float(np.min(stance_mask)) < 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0"))
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0"))
            # v962: 前轮 SWING 期 QP yaw 阻尼减半以下(-20->-6)——侧向力预
            # 算留给 pitch/roll，yaw 交给支撑轮差速(见轮层)。后轮爬顶期
            # (双前腿支撑)保持 -20。
            _ay_k = float(os.environ.get("S10_QP_AY_FRONT", "-6.0"))
            if float(getattr(self, '_rear_swing', 0.0)) > 0.5:
                _ay_k = float(os.environ.get("S10_QP_AY_K", "-20.0"))"""
assert old in src, "editA anchor missing"
src = src.replace(old, new)

# ---------- Edit B: all-stance wheel control uses nav omega ----------
old = """                else:
                    _tw = -(self.wheel_k * (self._vx_f - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))"""
new = """                else:
                    # v962: 全支撑(接近段)也执行导航 omega 差速——原仅跟 vx_f
                    # 导致进梯前 yaw 漂移 0.3rad+，首轮贴面不对称 → 自旋级联
                    _vref = (self._vx_f
                             - _side_s[leg] * self._om_f * self.track_half)
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))"""
assert old in src, "editB anchor missing"
src = src.replace(old, new)

# ---------- Edit C: any-swing stance wheels get yaw-rate differential ----------
old = """                if _any_swing:
                    # v942: 任何 swing 期支撑轮速度 PID（v958 开环满驱在前
                    # 轮爬升期自旋 3s 翻车回退）
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = (float(qvel[5]) * _kd_y
                                if float(np.max(step_lift[2:4])) > 0.5 else 0.0)
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half"""
new = """                if _any_swing:
                    # v942: 任何 swing 期支撑轮速度 PID（v958 开环满驱在前
                    # 轮爬升期自旋 3s 翻车回退）
                    # v962: 差速反馈任意 swing 期启用(原仅后轮爬顶)——前轮
                    # SWING 期后轴支撑轮用 yaw_rate 差速纠偏，释放 QP 侧向力
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = float(qvel[5]) * _kd_y
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half"""
assert old in src, "editC anchor missing"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")