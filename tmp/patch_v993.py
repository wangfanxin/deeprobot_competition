#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                if _any_swing:
                    # v942: 任何 swing 期支撑轮速度 PID（v958 开环满驱在前
                    # 轮爬升期自旋 3s 翻车回退）
                    # v992: 差速反馈任意 swing 期启用——前轮抬空后失去 yaw
                    # 阻力，后轮任何不对称都会让狗转圈(v991 宽窗+动量配置
                    # yaw 1.3→-0.11 自旋翻车实测)；支撑轮差速主动抗旋。
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = float(qvel[5]) * _kd_y
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))"""
new = """                if _any_swing:
                    # v942: 任何 swing 期支撑轮速度 PID（v958 开环满驱在前
                    # 轮爬升期自旋 3s 翻车回退）
                    # v992: 差速反馈任意 swing 期启用——前轮抬空后失去 yaw
                    # 阻力，后轮任何不对称都会让狗转圈(v991 宽窗+动量配置
                    # yaw 1.3→-0.11 自旋翻车实测)；支撑轮差速主动抗旋。
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = float(qvel[5]) * _kd_y
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    # v993: 支撑轮前驱下限——狗撞棱 body 停、后轮空转超速，
                    # 速度 PID 全力刹车(正力矩=后退)把狗夹死在棱口(实测
                    # tauW=+13.5 后轮后退、vx_w≈0 死锁 8s)。SWING 期支撑轮
                    # 至少保持 -DRIVE_FLOOR 前驱(负=前)，取 max(PID, floor)。
                    _df = -float(os.environ.get("S10_QP_DRIVE_FLOOR", "6.0"))
                    _tw = max(float(_tw), _df)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")