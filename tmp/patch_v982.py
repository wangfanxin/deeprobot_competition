#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        for leg in range(4):"""
new = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        # v982: 贴面区强制前驱(在 v981 接近段对准之后重试)——v968/980 自旋
        # 的根因是进梯前 yaw 漂移 0.2-0.5rad、双轮贴面不对称；现在接近段
        # 执行导航 omega，狗对准上棱，贴面轮给至少 -FACE_DRIVE 前驱滚上
        # 立面(否则轮被面卡住、swing 抬升只把轮压死在面上)
        _face_drive = np.zeros(4, dtype=bool)
        try:
            _fd_lo = float(os.environ.get("S10_QP_FACE_DRIVE_LO", "-0.12"))
            _fd_hi = float(os.environ.get("S10_QP_FACE_DRIVE_HI", "0.05"))
            for _fl in range(4):
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    if _dhv <= 0.085:
                        continue
                    _ddw = float(np.dot(wheel_xyz[_fl, :2] - _rp, _tng))
                    if _fd_lo < _ddw < _fd_hi:
                        _face_drive[_fl] = True
                        break
        except Exception:
            pass
        _fd_tau = -float(os.environ.get("S10_QP_FACE_DRIVE", "8.0"))
        for leg in range(4):"""
assert old in src, "edit1"
src = src.replace(old, new)

old = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
                else:
                    # v981: 全支撑(接近段)执行导航 omega 差速——原仅跟 vx_f
                    # 导致进梯前 yaw 漂移 0.2-0.5rad，首轮贴面不对称→自旋
                    _vref = (self._vx_f
                             - _side_s[leg] * self._om_f * self.track_half)
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = -1.5"""
new = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    if _face_drive[leg]:
                        _tw = min(float(_tw), _fd_tau)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
                else:
                    # v981: 全支撑(接近段)执行导航 omega 差速——原仅跟 vx_f
                    # 导致进梯前 yaw 漂移 0.2-0.5rad，首轮贴面不对称→自旋
                    _vref = (self._vx_f
                             - _side_s[leg] * self._om_f * self.track_half)
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    if _face_drive[leg]:
                        _tw = min(float(_tw), _fd_tau)
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = (_fd_tau if _face_drive[leg]
                                         else -1.5)"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")