#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# init rate-limit tracker
old = """        self._ov_sw_f = np.zeros(4)
        self._ov_st_f = np.zeros(4)
        self._osqp = None"""
new = """        self._ov_sw_f = np.zeros(4)
        self._ov_st_f = np.zeros(4)
        # v984: swing 目标升速限制跟踪（防 PD 过冲悬空）
        self._sw_zt = None
        self._osqp = None"""
assert old in src, "edit0"
src = src.replace(old, new)

# reset tracker at swing trigger
old = """                    self._sp[i] = 1.0
                    self._sp_top[i] = top[i]
                    self._rel_t[i] = None
                    self._sw_t0[i] = self._t"""
new = """                    self._sp[i] = 1.0
                    self._sp_top[i] = top[i]
                    self._rel_t[i] = None
                    self._sw_t0[i] = self._t
                    # v984: 触发时目标跟踪从当前轮高开始
                    if getattr(self, '_sw_zt', None) is None:
                        self._sw_zt = np.zeros(4)
                    self._sw_zt[i] = float(wz[i])"""
assert old in src, "edit1"
src = src.replace(old, new)

# apply rate limit in swing branch
old = """                # v964: 记录抬升目标轮高（QP 外力矩建模用）
                if getattr(self, '_sw_z_tgt', None) is None:
                    self._sw_z_tgt = np.zeros(4)
                self._sw_z_tgt[leg] = _wz_t"""
new = """                # v964: 记录抬升目标轮高（QP 外力矩建模用）
                if getattr(self, '_sw_z_tgt', None) is None:
                    self._sw_z_tgt = np.zeros(4)
                self._sw_z_tgt[leg] = _wz_t
                # v984: 目标升速限制——贴面轮廓随前速变化可达 3m/s，PD 追
                # 不上→滞后积累后过冲(轮悬空 0.75-0.9 无抓地实测)。目标只
                # 允许升 SW_TGT_RATE(默认1.2m/s)，轮靠贴面接触自然上升，
                # PD 只做引导；目标可自由下降(取 min)。
                _sw_rate = float(os.environ.get("S10_QP_SW_TGT_RATE", "1.2"))
                if getattr(self, '_sw_zt', None) is None:
                    self._sw_zt = np.zeros(4)
                _wz_t = min(_wz_t, self._sw_zt[leg] + _sw_rate * dt)
                self._sw_zt[leg] = _wz_t"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")