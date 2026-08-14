#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# 1) face_drive computation before leg loop
old = """        self._step_lift_last = step_lift.copy()"""
new = """        self._step_lift_last = step_lift.copy()
        # v997: 贴面区判定(软跟随与弱前驱共用)
        _face_drive = np.zeros(4, dtype=bool)
        try:
            _fd_lo = float(os.environ.get("S10_QP_FACE_DRIVE_LO", "-0.12"))
            _fd_hi = float(os.environ.get("S10_QP_FACE_DRIVE_HI", "0.05"))
            for _fl in range(4):
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    _dhmin = float(os.environ.get(
                        "S10_STAIR_RISER_MIN", "0.085"))
                    if _dhv <= _dhmin:
                        continue
                    _ddw = float(np.dot(wheel_xyz[_fl, :2] - _rp, _tng))
                    if _fd_lo < _ddw < _fd_hi:
                        _face_drive[_fl] = True
                        break
        except Exception:
            pass
        _fd_tau = -float(os.environ.get("S10_QP_FACE_DRIVE", "4.0"))"""
assert old in src, "edit1"
src = src.replace(old, new)

# 2) swing: soft follow in face zone (override gains and target)
old = """                # v934: 前后轴抬升增益不对称——前轮爬升有动量辅助用软增益
                # （防过伸/泵高）；后轮爬顶需主动抬升 0.125m 用硬增益
                _kps_d = float(os.environ.get("S10_QP_KP_SW", str(self.kp)))
                if leg in (2, 3):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_REAR",
                                                  str(self.kp)))
                # v935: swing kd 默认 30（原 8 阻尼比 ~0.25 欠阻尼，轮离地
                # 后自由过冲到 1.1 悬空实测）
                _kds_d = float(os.environ.get("S10_QP_KD_SW", "30.0"))"""
new = """                # v934: 前后轴抬升增益不对称——前轮爬升有动量辅助用软增益
                # （防过伸/泵高）；后轮爬顶需主动抬升 0.125m 用硬增益
                _kps_d = float(os.environ.get("S10_QP_KP_SW", str(self.kp)))
                if leg in (2, 3):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_REAR",
                                                  str(self.kp)))
                # v935: swing kd 默认 30（原 8 阻尼比 ~0.25 欠阻尼，轮离地
                # 后自由过冲到 1.1 悬空实测）
                _kds_d = float(os.environ.get("S10_QP_KD_SW", "30.0"))
                # v997: 贴面软阻尼跟随——前轮贴面时 KP 降到 15、KD 提到 100，
                # 目标跟轮高+3mm，轮靠前驱滚上立面，腿只吸收冲击(位置 PD
                # 硬抬让轮悬空无抓地、硬跟随+强前驱爆炸，均实测失败)。
                if _face_drive[leg] and leg in (0, 1):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_SOFT", "15.0"))
                    _kds_d = float(os.environ.get("S10_QP_KD_SW_SOFT", "100.0"))
                    _wz_t = float(wheel_xyz[leg, 2]) + float(os.environ.get(
                        "S10_QP_FOLLOW_GAP", "0.003"))"""
assert old in src, "edit2"
src = src.replace(old, new)

# 3) face drive in wheel loop
old = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    # v993: 支撑轮前驱下限"""
new = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    if _face_drive[leg]:
                        _tw = min(float(_tw), _fd_tau)
                    # v993: 支撑轮前驱下限"""
assert old in src, "edit3"
src = src.replace(old, new)

# swing wheel torque: face drive
old = """            else:
                tau[WHEEL_Q_IDX[leg]] = -1.5"""
new = """            else:
                tau[WHEEL_Q_IDX[leg]] = (_fd_tau if _face_drive[leg]
                                         else -1.5)"""
assert old in src, "edit4"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")