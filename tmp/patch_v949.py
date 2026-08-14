# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """    def _update_phases(self, body_pos, fwd, wheel_xyz):
        _fax = body_pos[:2] + fwd * 0.228
        _rax = body_pos[:2] - fwd * 0.228
        _df, _tf = self._nearest_riser(_fax)
        _dr, _tr = self._nearest_riser(_rax)
        _wz_f = float(np.mean([wheel_xyz[i, 2] for i in (0, 1)]))
        _wz_r = float(np.mean([wheel_xyz[i, 2] for i in (2, 3)]))
        r = self.fk.r
        _swd = float(os.environ.get("S10_STAIR_SWING_D", "0.30"))
        _to = float(os.environ.get("S10_STAIR_SWING_TO", "1.5"))
        if self._sp_f <= 0.0:
            # v946: 爬完锁存——前轮一旦成功到台面高度(top+r+0.005)且过棱，
            # 该 riser 不再重触发 SWING（狗在棱口晃动 d 振荡反复重触发、
            # 轮一直悬空不推进实测）；远离该 riser(d<-0.5)才复位
            if not hasattr(self, "_sp_f_done"):
                self._sp_f_done = False
            if _df < -0.5:
                self._sp_f_done = False
            if (not self._sp_f_done and -_swd < _df < 0.05
                    and self._sp_r <= 0.0):
                self._sp_f = 1.0
                self._sp_f_top = _tf
                self._rel_f_t = None
                self._sw_f_t0 = self._t
        else:
            # v935: 释放阈值 0.10（滞回）——触发 <-0.30..0.05，释放 >0.10。
            # v943 试 0.08 引发 yaw/roll 新级联（5s 翻车）回退；0.10 状态
            # 前轮上台面稳定 15s（roll±0.5），卡在悬空无法推进。
            if (_df > 0.10 and _wz_f >= self._sp_f_top + r + 0.005
                    and not getattr(self, '_sp_f_done', False)):
                self._sp_f_done = True
                if self._rel_f_t is None:
                    self._rel_f_t = self._t
                elif self._t - self._rel_f_t >= 0.05:
                    self._sp_f = 0.0
                    self._rel_f_t = None
            else:
                self._rel_f_t = None
            if self._t - self._sw_f_t0 > _to:
                self._sp_f = 0.0
                self._rel_f_t = None
        if self._sp_r <= 0.0:
            if -_swd < _dr < 0.05 and self._sp_f <= 0.0:
                self._sp_r = 1.0
                self._sp_r_top = _tr
                self._rel_r_t = None
                self._sw_r_t0 = self._t
        else:
            if _dr > 0.10 and _wz_r >= self._sp_r_top + r + 0.005:
                if self._rel_r_t is None:
                    self._rel_r_t = self._t
                elif self._t - self._rel_r_t >= 0.05:
                    self._sp_r = 0.0
                    self._rel_r_t = None
            else:
                self._rel_r_t = None
            if self._t - self._sw_r_t0 > _to:
                self._sp_r = 0.0
                self._rel_r_t = None
        step_lift = np.array([self._sp_f, self._sp_f,
                              self._sp_r, self._sp_r], dtype=np.float64)
        place_z = np.array([(self._sp_f_top if self._sp_f > 0 else 0.0)] * 2
                           + [(self._sp_r_top if self._sp_r > 0 else 0.0)] * 2,
                           dtype=np.float64)
        return step_lift, place_z"""
assert old in src
new = """    # v949: 单轮序列 ModeSchedule——FL->FR->RL->RR 依次爬升，任意时刻
    # >=3 点支撑（消除 2 点支撑 roll 崩）。每轮独立相位：
    #   触发: d_i 进入窗内（FL/RL 提前 0.30，FR/RR 等对侧爬完）
    #   释放: d_i>0.08 且轮高>=台面顶+r
    #   序列约束: FR 等 FL 完成; RL 等前轴完成; RR 等 RL 完成
    def _update_phases(self, body_pos, fwd, wheel_xyz):
        r = self.fk.r
        _swd = float(os.environ.get("S10_STAIR_SWING_D", "0.30"))
        _to = float(os.environ.get("S10_STAIR_SWING_TO", "1.5"))
        if not hasattr(self, "_sp"):
            self._sp = np.zeros(4)
            self._sp_top = np.zeros(4)
            self._sw_t0 = np.full(4, -1e9)
            self._rel_t = [None] * 4
            self._done = np.zeros(4, dtype=bool)
        d = np.zeros(4); top = np.zeros(4)
        for i in range(4):
            d[i], top[i] = self._nearest_riser(wheel_xyz[i, :2])
        wz = wheel_xyz[:, 2]
        # 远离当前 riser 时复位完成标志（下一级）
        for i in range(4):
            if d[i] < -0.5:
                self._done[i] = False
        # 触发/释放
        for i in range(4):
            _lead = (i in (0, 2))           # FL/RL 提前触发
            _opp_done = self._done[i ^ 1]   # 对侧轮完成（FR 等 FL, RR 等 RL）
            _front_done = bool(np.all(self._done[0:2])) if i >= 2 else True
            if self._sp[i] <= 0.0:
                _win_lo = -_swd if _lead else -0.05
                _win_hi = 0.05 if _lead else 0.10
                _ok = (_win_lo < d[i] < _win_hi
                       and (self._done[i] is False)
                       and (not _lead or _front_done)
                       and (_lead or _opp_done))
                if _ok:
                    self._sp[i] = 1.0
                    self._sp_top[i] = top[i]
                    self._rel_t[i] = None
                    self._sw_t0[i] = self._t
            else:
                if (d[i] > 0.08 and wz[i] >= self._sp_top[i] + r + 0.005
                        and not self._done[i]):
                    self._done[i] = True
                    if self._rel_t[i] is None:
                        self._rel_t[i] = self._t
                    elif self._t - self._rel_t[i] >= 0.05:
                        self._sp[i] = 0.0
                        self._rel_t[i] = None
                else:
                    self._rel_t[i] = None
                if self._t - self._sw_t0[i] > _to:
                    self._sp[i] = 0.0
                    self._rel_t[i] = None
        step_lift = self._sp.copy()
        place_z = np.array([self._sp_top[i] if self._sp[i] > 0 else 0.0
                            for i in range(4)], dtype=np.float64)
        return step_lift, place_z"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v949 modeschedule")