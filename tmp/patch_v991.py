#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# 1) face drive block re-add (before wheel loop)
old = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        for leg in range(4):"""
new = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        # v991: 贴面区强制前驱——轮滚上立面(0.125m 台阶>轮半径 0.081，纯腿
        # 抬升悬空、纯动量滚不上均实测失败；轮驱+腿跟随是唯一可行机制)。
        # 轴式调度对称爬升，双轮一起滚。
        _face_drive = np.zeros(4, dtype=bool)
        try:
            _fd_lo = float(os.environ.get("S10_QP_FACE_DRIVE_LO", "-0.12"))
            _fd_hi = float(os.environ.get("S10_QP_FACE_DRIVE_HI", "0.05"))
            for _fl in range(4):
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    _dhmin = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
                    if _dhv <= _dhmin:
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

# 2) face drive usage
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

# 3) swing target: follow wheel in face zone (override face profile)
old = """                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t"""
new = """                # v991: 贴面跟随——目标=轮子实际高度+5mm 轻引导(不硬抬)。
                # 之前贴面轮廓绝对目标把轮拉到 0.747 悬空；跟随目标让轮靠
                # 前驱沿立面滚上，腿只保证轮子不塌不跳。
                if _face_drive[leg]:
                    _wz_t = float(wheel_xyz[leg, 2]) + float(os.environ.get(
                        "S10_QP_FOLLOW_GAP", "0.005"))
                _wz_act2 = float(wheel_xyz[leg, 2])
                _over2 = _wz_act2 - _wz_t"""
assert old in src, "edit3"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")