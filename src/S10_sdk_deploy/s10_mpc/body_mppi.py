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
纯 numpy，N=4096/H=40/dt=0.05 → 2.0s 视界（用户 2026-08-10：进弯刹车交给
MPPI 摩擦锥 + 长视界，导航不再做动力学限速）；20Hz 调用（CPU）。
"""
import numpy as np
from s10_mpc.vmc_legs import car_omega_limit


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class BodyMPPI:
    def __init__(self, N=4096, H=40, dt=0.05,
                 tau_v=0.15, tau_w=0.10, mu=0.75, g=9.81,
                 vx_max=6.0, omega_max=4.0,
                 lam=0.35, sigma_vx=0.45, sigma_om=0.55,
                 w_dist=2.0, w_v=0.80, w_h=0.0, w_s=0.05,
                 ada_alpha=0.35, ada_min=0.5, ada_max=2.0,
                 seed=0, a_max=2.5):
        # v427: 精简参数集（Tube-MPPI 灵感落地的前提：调参面越小，
        # 执行层跟踪管越可信）。只保留启动脚本实际调用的旋钮：
        #   S10_MPPI_MU      —— 摩擦锥 μ（标定 μ_eff≈0.36）
        #   S10_MPPI_OMAX    —— 转向上限（由 VMC 执行能力派生）
        #   S10_MPPI_W_GUIDE —— 指令跟踪权重（管外拉回力度）
        #   S10_MPPI_W_DIST  —— 路径距离成本权重
        # N/H/dt/W_V/W_HEAD/W_S/VMAX 固定默认（防参数爆炸）；控制量上限仍由
        # VMC 派生：S10_VMC_OM_CAP/ABS_MAX、S10_AUTO_VMAX。
        import os as _os
        mu = float(_os.environ.get('S10_MPPI_MU', mu))
        omega_max = float(_os.environ.get('S10_MPPI_OMAX', omega_max))
        _vmc_om = _os.environ.get('S10_VMC_OM_CAP') or _os.environ.get(
            'S10_VMC_OM_ABS_MAX')
        if _vmc_om:
            omega_max = min(omega_max, float(_vmc_om))
        _vmc_vx = _os.environ.get('S10_AUTO_VMAX')
        if _vmc_vx:
            vx_max = min(vx_max, float(_vmc_vx))
        w_dist = float(_os.environ.get('S10_MPPI_W_DIST', w_dist))
        w_g = float(_os.environ.get('S10_MPPI_W_GUIDE', '0.5'))
        # v435: 距离成本对准衰减——狗当前航向与参考航向误差大时（离线/
        # 转向中），距离成本整体淡出，让 guide 指令主导（实测 wp3→4 狗偏
        # 北 1.5m 时 w_dist*d_min 压过 w_g，MPPI 输出 om≈0 不转向绕圈）；
        # 对准后距离成本恢复精修路线。连续量，无门控。
        self.omega_lim_fn = car_omega_limit
        self.N, self.H, self.dt = N, H, dt
        self.tau_v, self.tau_w = tau_v, tau_w
        self.mu, self.g = mu, g
        self.vx_max, self.omega_max = vx_max, omega_max
        # v826: 纵向加速度硬约束（S10_MPPI_A_MAX，默认 2.5 m/s² =
        # CarVMC 能力标定 0→5m/s≈2s）。起步/出弯不允许 MPPI 直接给出
        # vref 阶跃指令（起步轮打滑→差速正反馈自旋侧翻实测，
        # run_v825final VMAX=6）。rollout 与输出同时钳制——规划模型
        # 与真实执行能力一致，AutoNavFollower 保持纯 xy 无动力学约束。
        self.a_max = float(_os.environ.get('S10_MPPI_A_MAX', a_max))
        self.lam = lam
        self.sigma_vx0, self.sigma_om0 = sigma_vx, sigma_om
        self.w_dist, self.w_v, self.w_h, self.w_s = w_dist, w_v, w_h, w_s
        self.w_g = w_g
        self.ada_alpha, self.ada_min, self.ada_max = ada_alpha, ada_min, ada_max
        # v506: DBaS 自适应 sigma 可调（S10_MPPI_ADA）——N=2048 时成本
        # 均值估计噪声大，sigma_scale 震荡导致起步（门架刀刃）失稳翻车。
        # 0=固定 sigma（最稳），0~1=缩放自适应增益，1=原行为。
        self.ada_alpha = self.ada_alpha * float(_os.environ.get(
            'S10_MPPI_ADA', '1'))
        self.rng = np.random.default_rng(seed)
        self._u = np.zeros(2)
        # v826b: 输出速率限幅基准——上一拍实际输出（非实测速度）。
        # 实测速度基准在起步/停顿时死锁（狗没动→指令永远 0.125m/s，
        # 轮矩<静摩擦卡死实测）；指令基准使 vx 确定性爬升，轮速误差
        # 增长→力矩突破静摩擦正常起步。
        self._out_prev = np.zeros(2)
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
            om_lim = np.minimum(om_lim, self.omega_lim_fn(vx_now))
            om_c = np.clip(om_c, -om_lim, om_lim)
            vx_c = np.clip(vx_c, 0.0, self.vx_max)
            # v826: 加速度硬约束（能力表派生，S10_MPPI_A_MAX）
            vx_c = np.clip(vx_c,
                           vx_now - self.a_max * dt,
                           vx_now + self.a_max * dt)
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

    def _cost(self, s, u_seq, prev_u, ref, v_ref, guide=None):
        """ref: (R,3) [x, y, heading] 路径参考轨迹；v_ref: 标量限速；
        guide: (2,) 导航期望 [vx, om]——v376 指令跟踪：MPPI 在执行层
        约束内跟随导航转向，距离成本只做辅助（wp1 后 donut 实测）。"""
        R = ref.shape[0]
        xy = s[:, :, 0:2]                               # (N,H+1,2)
        # v435: 用当前狗状态（s0）相对最近参考点的航向误差算对准系数
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
        if guide is not None and self.w_g > 0.0:
            cost += self.w_g * np.sum(
                (u_seq - np.asarray(guide, dtype=np.float64)[None, None, :])
                ** 2, axis=(1, 2))
        cost += self.w_s * np.sum((u_seq - prev_u[None, None, :]) ** 2,
                                  axis=(1, 2))
        return cost

    def plan(self, state, ref, v_ref, prev_u=None, guide_om=None):
        """state: (x,y,yaw,vx,vy,ω)；ref: (R,3) 路径参考轨迹；v_ref: 限速；
        guide_om: 可选曲率前馈（κ·v_ref）作采样中心——默认用参考航向变化率。
        """
        s0 = np.asarray(state, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        prev_u = np.asarray(prev_u if prev_u is not None else self._u,
                            dtype=np.float64)
        sv = self.sigma_vx0 * self.sigma_scale
        so = self.sigma_om0 * self.sigma_scale
        # v270: 采样中心 = v_ref + 曲率前馈 κ·v（赛用摩托恒定转向率），
        # 约束仍在 _rollout 摩擦锥内；默认用参考航向变化率兜底。
        guide_vx = float(np.clip(v_ref, 0.0, self.vx_max))
        if guide_om is None:
            guide_om = 0.0
            if ref.shape[0] >= 3:
                _dh = _wrap(float(ref[2, 2]) - float(ref[0, 2]))
                guide_om = float(np.clip(
                    _dh * guide_vx / 2.0, -self.omega_max, self.omega_max))
        else:
            _om_cap = float(np.minimum(
                self.omega_max, self.omega_lim_fn(s0[3])))
            guide_om = float(np.clip(guide_om, -_om_cap, _om_cap))
        noise_vx = self.rng.normal(0.0, sv, (self.N, self.H))
        noise_om = self.rng.normal(0.0, so, (self.N, self.H))
        u_seq = np.zeros((self.N, self.H, 2))
        u_seq[:, :, 0] = guide_vx + noise_vx
        u_seq[:, :, 1] = guide_om + noise_om
        # v349: 采样不硬钳（钳制会改变采样分布触发直道混沌失稳 v346-348 实测）；
        # 控制上限由 VMC 派生的 omega_max/vx_max 在输出钳制与 rollout 摩擦锥
        # 中强制执行（软实现：超限样本经摩擦锥/输出钳制吸收）。
        u_seq[0] = prev_u
        s = self._rollout(s0, u_seq, prev_u)
        cost = self._cost(s, u_seq, prev_u, ref, v_ref,
                          guide=[guide_vx, guide_om])
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
        # v314: 输出钳制——u_new 是未钳制样本的加权平均，会超过 vx_max/
        # omega_max（起步 3.65 m/s 实测在缓坡上触发 VMC 轮层自旋 om -4.37）。
        # rollout 内的钳制只作用于动力学，最终指令必须落在执行包线内。
        # v318: vx 额外钳到当前 v_ref——MPPI 加权平均会因路径距离收益
        # 系统性超速 0.2-0.3 m/s（坡顶 3.5 vs vlim 3.23 过脊离地自旋实测）。
        _vcap = min(self.vx_max, guide_vx)
        # v826c: 输出 omega 钳制与 rollout 同一套约束（此前漏了摩擦锥
        # μg/|v| 与最小转弯半径——vx≈2.5 时输出 ω=-2.0 → a_lat=5.0
        # 超侧翻包线，wp4→5 发卡翻车实测）。
        _v_abs = max(abs(float(s0[3])), 1e-3)
        _om_out = float(np.minimum(
            self.omega_max,
            min(self.mu * self.g / _v_abs,
                float(self.omega_lim_fn(s0[3])))))
        # v826b: 输出加速度钳制——以**上一拍输出指令**为基准做速率
        # 限幅（vx 变化 ≤ a_max·dt）。起步指令确定性爬升，突破静摩擦
        # 正常起步；不会像 v826a 用实测速度那样死锁，也不会像无钳制
        # 那样 0→6 阶跃打滑自旋侧翻。
        _vx_out = float(np.clip(
            u_new[0],
            self._out_prev[0] - self.a_max * self.dt,
            self._out_prev[0] + self.a_max * self.dt))
        _vx_out = float(np.clip(_vx_out, 0.0, _vcap))
        u_out = np.array([_vx_out,
                          np.clip(u_new[1], -_om_out, _om_out)])
        self._out_prev = u_out
        return float(u_out[0]), float(u_out[1])
