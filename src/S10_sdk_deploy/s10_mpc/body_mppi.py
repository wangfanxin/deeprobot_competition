"""body_mppi.py — v218 方案规划层：身体层 MPPI（[vx, ω] 2 维控制）。

6 状态模型：s = [x, y, yaw, vx, vy, ω]
  yaw += ω·dt
  x   += (vx·cos yaw − vy·sin yaw)·dt
  y   += (vx·sin yaw + vy·cos yaw)·dt
  vx  += (vx_cmd − vx)·dt/τv
  vy  += (0 − vy)·dt/τv
  ω   += (ω_cmd − ω)·dt/τω
摩擦锥硬约束：|vx·ω| ≤ μ·g（采样后 clamp ω），速度越高转向上限越小。
成本：到目标点距离 + 速度偏差 + 航向偏差 + 控制平滑；softmax 加权平均更新。
DBaS 自适应 σ：成本高时放大采样噪声，低时收敛。
纯 numpy，N=256/H=20/dt=0.05 → 单次 plan <5ms，可跑 20-30Hz（CPU）。
"""
import numpy as np


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class BodyMPPI:
    def __init__(self, N=256, H=20, dt=0.05,
                 tau_v=0.15, tau_w=0.10, mu=0.75, g=9.81,
                 vx_max=5.0, omega_max=3.5,
                 lam=0.35, sigma_vx=0.45, sigma_om=0.55,
                 w_dist=1.0, w_v=0.30, w_h=0.8, w_s=0.15,
                 ada_alpha=0.35, ada_min=0.5, ada_max=2.0,
                 seed=0):
        # v269: 关键参数开放环境变量（μ 需匹配实测 a_lat 包线：
        # a_lat=μg，CarVMC 实测 3.5 -> μ_eff≈0.36）
        import os as _os
        N = int(_os.environ.get('S10_MPPI_N', N))
        H = int(_os.environ.get('S10_MPPI_H', H))
        dt = float(_os.environ.get('S10_MPPI_DT', dt))
        mu = float(_os.environ.get('S10_MPPI_MU', mu))
        omega_max = float(_os.environ.get('S10_MPPI_OMAX', omega_max))
        vx_max = float(_os.environ.get('S10_MPPI_VMAX', vx_max))
        self.N, self.H, self.dt = N, H, dt
        self.tau_v, self.tau_w = tau_v, tau_w
        self.mu, self.g = mu, g
        self.vx_max, self.omega_max = vx_max, omega_max
        self.lam = lam
        self.sigma_vx0, self.sigma_om0 = sigma_vx, sigma_om
        self.w_dist, self.w_v, self.w_h, self.w_s = w_dist, w_v, w_h, w_s
        self.ada_alpha, self.ada_min, self.ada_max = ada_alpha, ada_min, ada_max
        self.rng = np.random.default_rng(seed)
        self._u = np.zeros(2)
        self._cost_ref = 1.0
        self.sigma_scale = 1.0

    def _rollout(self, s0, u_seq, prev_u):
        """u_seq: (N, H, 2) 控制序列 -> states (N, H+1, 6)。"""
        N, H, dt = self.N, self.H, self.dt
        s = np.broadcast_to(s0[None, None, :], (N, H + 1, 6)).copy()
        for h in range(H):
            vx_c = u_seq[:, h, 0]
            om_c = u_seq[:, h, 1]
            # 摩擦锥 clamp：|vx·ω| ≤ μ·g
            vx_now = s[:, h, 3]
            om_lim = np.minimum(self.omega_max,
                                self.mu * self.g / (np.abs(vx_now) + 1e-3))
            om_c = np.clip(om_c, -om_lim, om_lim)
            vx_c = np.clip(vx_c, 0.0, self.vx_max)
            yaw = s[:, h, 2]
            s[:, h + 1, 2] = yaw + om_c * dt
            s[:, h + 1, 0] = s[:, h, 0] + (vx_now * np.cos(yaw)
                                           - s[:, h, 4] * np.sin(yaw)) * dt
            s[:, h + 1, 1] = s[:, h, 1] + (vx_now * np.sin(yaw)
                                           + s[:, h, 4] * np.cos(yaw)) * dt
            s[:, h + 1, 3] = vx_now + (vx_c - vx_now) * dt / self.tau_v
            s[:, h + 1, 4] = s[:, h, 4] + (0.0 - s[:, h, 4]) * dt / self.tau_v
            s[:, h + 1, 5] = s[:, h, 5] + (om_c - s[:, h, 5]) * dt / self.tau_w
        return s

    def _cost(self, s, u_seq, prev_u, ref, v_ref):
        """ref: (R,3) [x, y, heading] 路径参考轨迹；v_ref: 标量限速。"""
        R = ref.shape[0]
        xy = s[:, :, 0:2]                               # (N,H+1,2)
        d2 = np.sum((xy[:, :, None, :] - ref[None, None, :, 0:2]) ** 2,
                    axis=-1)                             # (N,H+1,R)
        i_min = np.argmin(d2, axis=-1)                  # (N,H+1)
        d_min = np.sqrt(np.take_along_axis(
            d2, i_min[..., None], axis=-1)[..., 0])
        h_ref = np.take_along_axis(
            np.broadcast_to(ref[None, None, :, 2], (self.N, self.H + 1, R)),
            i_min[..., None], axis=-1)[..., 0]
        h_err = _wrap(s[:, :, 2] - h_ref)
        v_err = s[:, :, 3] - v_ref
        cost = (self.w_dist * d_min
                + self.w_h * h_err ** 2
                + self.w_v * v_err ** 2)
        cost = cost.sum(axis=1)
        cost += self.w_s * np.sum((u_seq - prev_u[None, None, :]) ** 2,
                                  axis=(1, 2))
        return cost

    def plan(self, state, ref, v_ref, prev_u=None):
        """state: (x,y,yaw,vx,vy,ω)；ref: (R,3) 路径参考轨迹；v_ref: 限速。"""
        s0 = np.asarray(state, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        prev_u = np.asarray(prev_u if prev_u is not None else self._u,
                            dtype=np.float64)
        sv = self.sigma_vx0 * self.sigma_scale
        so = self.sigma_om0 * self.sigma_scale
        # v218: 围绕"标称控制"采样（v_ref + 参考航向变化率），保证样本覆盖
        # 低成本区（纯 prev±σ 在 σ 小时永远到不了 v_ref 处的最优）。
        guide_vx = float(np.clip(v_ref, 0.0, self.vx_max))
        guide_om = 0.0
        if ref.shape[0] >= 3:
            _dh = _wrap(float(ref[2, 2]) - float(ref[0, 2]))
            guide_om = float(np.clip(
                _dh * guide_vx / 2.0, -self.omega_max, self.omega_max))
        noise_vx = self.rng.normal(0.0, sv, (self.N, self.H))
        noise_om = self.rng.normal(0.0, so, (self.N, self.H))
        u_seq = np.zeros((self.N, self.H, 2))
        u_seq[:, :, 0] = guide_vx + noise_vx
        u_seq[:, :, 1] = guide_om + noise_om
        u_seq[0] = prev_u
        s = self._rollout(s0, u_seq, prev_u)
        cost = self._cost(s, u_seq, prev_u, ref, v_ref)
        cmin = float(cost.min())
        w = np.exp(-(cost - cmin) / max(self.lam, 1e-6))
        w = w / (w.sum() + 1e-9)
        u_new = np.sum(w[:, None, None] * u_seq, axis=0)[0]
        self._u = u_new
        cm = float(cost.mean())
        self._cost_ref = 0.9 * self._cost_ref + 0.1 * max(cm, 1e-3)
        self.sigma_scale = float(np.clip(
            1.0 + self.ada_alpha * (cm - self._cost_ref)
            / max(self._cost_ref, 1e-3),
            self.ada_min, self.ada_max))
        return u_new[0], u_new[1]
