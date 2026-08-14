#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# ---------- Edit 1: init cool-down array ----------
old = """        if not hasattr(self, "_sp"):
            self._sp = np.zeros(4)
            self._sp_top = np.zeros(4)
            self._sw_t0 = np.full(4, -1e9)
            self._rel_t = [None] * 4
            self._done = np.zeros(4, dtype=bool)"""
new = """        if not hasattr(self, "_sp"):
            self._sp = np.zeros(4)
            self._sp_top = np.zeros(4)
            self._sw_t0 = np.full(4, -1e9)
            self._rel_t = [None] * 4
            self._done = np.zeros(4, dtype=bool)
            # v963: swing 超时冷却——避免棱口无限"超时->重触发"循环
            self._sw_cd = np.full(4, -1e9)"""
assert old in src, "edit1 anchor missing"
src = src.replace(old, new)

# ---------- Edit 2: trigger adds cool-down requirement ----------
old = """                _ok = (_win_lo < d[i] < _win_hi
                       and not self._done[i]
                       and (not _lead or _front_done)
                       and (_lead or _opp_done)
                       and _roll_gate)"""
new = """                _ok = (_win_lo < d[i] < _win_hi
                       and not self._done[i]
                       and (not _lead or _front_done)
                       and (_lead or _opp_done)
                       and _roll_gate
                       and self._t - self._sw_cd[i] > 0.5)"""
assert old in src, "edit2 anchor missing"
src = src.replace(old, new)

# ---------- Edit 3: release condition relaxed to d>0.0 ----------
old = """            else:
                if (d[i] > 0.08 and wz[i] >= self._sp_top[i] + r + 0.005
                        and not self._done[i]):"""
new = """            else:
                # v963: 释放放宽到轮心过棱(d>0)且高度达标——原 d>0.08 在
                # 贴面窗内轮子悬在台面高度却永远等不到 d 推进(棱口卡死)
                if (d[i] > 0.0 and wz[i] >= self._sp_top[i] + r - 0.005
                        and not self._done[i]):"""
assert old in src, "edit3 anchor missing"
src = src.replace(old, new)

# ---------- Edit 4: timeout sets cool-down ----------
old = """                if self._t - self._sw_t0[i] > _to:
                    self._sp[i] = 0.0
                    self._rel_t[i] = None"""
new = """                if self._t - self._sw_t0[i] > _to:
                    self._sp[i] = 0.0
                    self._rel_t[i] = None
                    self._sw_cd[i] = self._t"""
assert old in src, "edit4 anchor missing"
src = src.replace(old, new)

# ---------- Edit 5: face-zone forward drive in wheel control ----------
old = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        for leg in range(4):
            _wq = float(qvel[WHEEL_QV_IDX[leg]])
            _vw = -_wq * self.fk.r
            if stance_mask[leg] > 0.5:"""
new = """        # 轮矩：支撑前驱、抬升 0（差速冻结，hip yaw 全程）
        _any_swing = float(np.max(step_lift)) > 0.5
        _side_s = np.array([-1.0, 1.0, -1.0, 1.0])
        # v963: 贴面区判断——轮心距最近高 riser 的水平投影 d∈[-0.18,0.05]
        # (棱口贴面/即将上沿)。贴面轮强制前驱(至少 -FACE_DRIVE)，防止轮
        # 被面带动空转后 PID 反刹 → 车身顶死在棱上(棱口卡死 8s 实测)。
        _face_drive = np.zeros(4, dtype=bool)
        try:
            _fd_lo = float(os.environ.get("S10_QP_FACE_DRIVE_LO", "-0.18"))
            _fd_hi = float(os.environ.get("S10_QP_FACE_DRIVE_HI", "0.05"))
            for leg in range(4):
                for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                    if _dhv <= 0.085:
                        continue
                    _ddw = float(np.dot(wheel_xyz[leg, :2] - _rp, _tng))
                    if _fd_lo < _ddw < _fd_hi:
                        _face_drive[leg] = True
                        break
        except Exception:
            pass
        _fd_tau = -float(os.environ.get("S10_QP_FACE_DRIVE", "8.0"))
        for leg in range(4):
            _wq = float(qvel[WHEEL_QV_IDX[leg]])
            _vw = -_wq * self.fk.r
            if stance_mask[leg] > 0.5:"""
assert old in src, "edit5 anchor missing"
src = src.replace(old, new)

# ---------- Edit 6: apply face drive inside stance branch ----------
old = """                if _any_swing:
                    # v942: 任何 swing 期支撑轮速度 PID（v958 开环满驱在前
                    # 轮爬升期自旋 3s 翻车回退）
                    # v962: 差速反馈任意 swing 期启用(原仅后轮爬顶)——前轮
                    # SWING 期后轴支撑轮用 yaw_rate 差速纠偏，释放 QP 侧向力
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = float(qvel[5]) * _kd_y
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
                else:
                    # v962: 全支撑(接近段)也执行导航 omega 差速——原仅跟 vx_f
                    # 导致进梯前 yaw 漂移 0.3rad+，首轮贴面不对称 → 自旋级联
                    _vref = (self._vx_f
                             - _side_s[leg] * self._om_f * self.track_half)
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                    tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                tau[WHEEL_Q_IDX[leg]] = -1.5"""
new = """                if _any_swing:
                    # v942: 任何 swing 期支撑轮速度 PID（v958 开环满驱在前
                    # 轮爬升期自旋 3s 翻车回退）
                    # v962: 差速反馈任意 swing 期启用(原仅后轮爬顶)——前轮
                    # SWING 期后轴支撑轮用 yaw_rate 差速纠偏，释放 QP 侧向力
                    _kd_y = float(os.environ.get("S10_QP_WHEEL_KD_Y", "3.0"))
                    _om_gain = float(qvel[5]) * _kd_y
                    _vref = self._vx_f - _side_s[leg] * _om_gain * self.track_half
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                else:
                    # v962: 全支撑(接近段)也执行导航 omega 差速——原仅跟 vx_f
                    # 导致进梯前 yaw 漂移 0.3rad+，首轮贴面不对称 → 自旋级联
                    _vref = (self._vx_f
                             - _side_s[leg] * self._om_f * self.track_half)
                    _tw = -(self.wheel_k * (_vref - _vw)) - self.wheel_d * _wq
                # v963: 贴面区强制前驱(取更向前的力矩)——防止 PID 反刹
                if _face_drive[leg]:
                    _tw = min(float(_tw), _fd_tau)
                tau[WHEEL_Q_IDX[leg]] = float(np.clip(_tw, -13.5, 13.5))
            else:
                # 抬升轮：贴面区用稍强前驱(防卡沿)，否则轻微前驱
                tau[WHEEL_Q_IDX[leg]] = (_fd_tau if _face_drive[leg]
                                         else -1.5)"""
assert old in src, "edit6 anchor missing"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")