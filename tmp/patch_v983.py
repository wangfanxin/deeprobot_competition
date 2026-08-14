#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# ---- remove face drive (harmful) ----
old = """        # v982: 贴面区强制前驱(在 v981 接近段对准之后重试)——v968/980 自旋
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
new = """        for leg in range(4):"""
assert old in src, "editA1"
src = src.replace(old, new)

old = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
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
new = """                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
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
assert old in src, "editA2"
src = src.replace(old, new)

# ---- axle-based ModeSchedule ----
old = """        # 触发/释放
        for i in range(4):
            _lead = (i in (0, 2))           # FL/RL 提前触发
            _opp_done = self._done[i ^ 1]   # 对侧轮完成（FR 等 FL, RR 等 RL）
            _front_done = bool(np.all(self._done[0:2])) if i >= 2 else True
            if self._sp[i] <= 0.0:
                _win_lo = -_swd if _lead else -0.05
                _win_hi = 0.05 if _lead else 0.10
                # v953: 对侧轮(FR/RR)触发要求 body 水平(|roll|<0.08)——FL
                # 爬完 body 侧滚、对侧髋变低必须折叠才能到台面 → 过伸实测
                _roll_gate = True
                if not _lead:
                    # v954: 门控放宽到 0.15——FL 上台面后 body 几何 roll 约
                    # 0.1（腿吸收后观测 -0.1），0.08 太紧 FR 永远等不到
                    _roll_gate = abs(getattr(self, '_body_roll', 0.0)) < 0.15
                _ok = (_win_lo < d[i] < _win_hi
                       and not self._done[i]
                       and (not _lead or _front_done)
                       and (_lead or _opp_done)
                       and _roll_gate
                       and self._t - self._sw_cd[i] > 0.5)"""
new = """        # v983: 轴式调度(用户批准方案)——前轴(FL+FR)成对触发、成对释放，
        # 后轴(RL+RR)等前轴全部 done 再触发。对称爬升消除单轮序列的贴面
        # 力不对称→自旋(v968/980/982 贴面前驱连续失败实测)。2 点支撑期
        # (前轴离地)roll 由 QP 差载控制(同轴两轮 y±0.181 可产生 roll 力矩)。
        for i in range(4):
            _front_axle = i in (0, 1)
            _rear_need = bool(np.all(self._done[0:2])) if not _front_axle else True
            if self._sp[i] <= 0.0:
                _win_lo = -_swd
                _win_hi = 0.05
                _ok = (_win_lo < d[i] < _win_hi
                       and not self._done[i]
                       and _rear_need
                       and self._t - self._sw_cd[i] > 0.5)"""
assert old in src, "editB1"
src = src.replace(old, new)

# ---- λ_ref: handle 2-point stance (axle swing) ----
old = """        _sw_l = [i for i in range(4) if qp_stance[i] <= 0.5]
        if len(_sw_l) == 1:
            _st_l = [i for i in range(4) if qp_stance[i] > 0.5]"""
new = """        _sw_l = [i for i in range(4) if qp_stance[i] <= 0.5]
        if len(_sw_l) == 2 and _sw_l in ([0, 1], [2, 3]):
            # v983: 轴式抬升 2 点支撑——同轴两轮均分 mg
            for _i in _sw_l:
                pass
            for _i in range(4):
                if qp_stance[_i] > 0.5:
                    lam_ref[_i, 2] = self.m * self.g / 2.0
        elif len(_sw_l) == 1:
            _st_l = [i for i in range(4) if qp_stance[i] > 0.5]"""
assert old in src, "editB2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")