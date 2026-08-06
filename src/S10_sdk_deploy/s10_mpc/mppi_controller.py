"""S10 轮足轻量 MPPI 控制器（模式 B 主控制器，替代 dial-mpc MBDPI）。

背景：
- dial-mpc MBDPI 早期每轮固定开销 ~2.4s（0.4Hz），且缺 shift 时 action 恒 0
  （2026-08-05 校正：retrace 修复后稳态 plan_once 实测 0.12~0.13s ≈ 8Hz；
  本文件为早期 MPPI 方案备份，现主控制器为 mpc_controller.py 的 MBDPI）
- 自写 MPPI（一次 jit rollout + softmax）H=30/64 样本 → 4.8Hz，P50 208ms
- 20-100Hz 的采样 MPC 在 S10 上不可达（Nsample×H 次 mjx step 计算极限）；
  现实方案：MPPI ~5Hz 轨迹规划 + 仿真 200Hz 轨迹执行（轮足慢速可用）

关键：
- **shift**（时间平移）让优化过的控制进入 Y[0]——缺 shift 则 action 恒 0
- H>=30（0.6s）才能"学动"（轮启动需 0.5s，短 horizon 内动=能量惩罚无收益）
"""
import threading
import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from dial_mpc.envs.s10_env import S10WheeledEnv, S10WheeledEnvConfig


class MPPIController:
    def __init__(self, yaml_path: str = None,
                 H: int = 30, Ns: int = 64, temp: float = 0.3,
                 sigma: float = 1.0, dt: float = 0.02,
                 smooth_penalty: float = 2.0):
        import yaml
        if yaml_path:
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            H = cfg.get("H", H)
            Ns = cfg.get("Nsample", Ns)
            temp = cfg.get("temp_sample", temp)
            sigma = cfg.get("sigma_scale", sigma)
            dt = cfg.get("dt", dt)
        self.H = H
        self.Ns = Ns
        self.temp = temp
        self.sigma = sigma
        self.dt = dt
        self.smooth_penalty = smooth_penalty

        print(f"[MPPI] 构建 env (dt={dt}) ...", flush=True)
        self.env = S10WheeledEnv(S10WheeledEnvConfig(
            dt=dt, timestep=dt, kp=40.0, kd=2.0, wheel_tau_scale=3.0))
        self.nu = self.env.action_size
        self._build_rollout()
        self.Y = jnp.zeros((H + 1, self.nu))
        self.rng = jax.random.PRNGKey(0)
        self.state = None
        self.cmd_vel = jnp.array([0.0, 0.0, 0.0])
        self.cmd_ang = jnp.array([0.0, 0.0, 0.0])
        self.ready = False      # init_state + 预编译完成后置 True
        # 异步规划
        self.latest_tau = np.zeros(self.nu, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _build_rollout(self):
        step = jax.jit(self.env.step)

        def rollout_scan(s0, us):
            def f(s, u):
                s2 = step(s, u)
                return s2, s2.reward
            _, rews = jax.lax.scan(f, s0, us)
            return rews

        self._rollout_vmap = jax.jit(
            jax.vmap(rollout_scan, in_axes=(None, 0)))

        @partial(jax.jit, static_argnums=(3, 4))
        def mppi(s0, Ym, rng, temp, sigma):
            eps = jax.random.normal(rng, (self.Ns, self.H + 1, self.nu))
            Y0s = jnp.clip(Ym + sigma * eps, -1.0, 1.0)
            Y0s = Y0s.at[:, 0].set(Ym[0])     # 第一个控制固定（当前执行）
            rews = self._rollout_vmap(s0, Y0s)
            rews = jnp.where(jnp.isnan(rews) | jnp.isinf(rews), -1e6, rews)
            # 控制平滑：罚样本偏离上一解（节点 1..H），抑制相邻规划方向翻转
            dev = jnp.mean(jnp.square(Y0s[:, 1:] - Ym[None, 1:]), axis=(1, 2))
            cost = rews.sum(axis=1) - self.smooth_penalty * dev
            w = jax.nn.softmax(cost / temp)
            return jnp.einsum("n,nij->ij", w, Y0s)

        self._mppi = mppi

        @jax.jit
        def shift(Ym):
            return jnp.roll(Ym, -1, axis=0).at[-1].set(jnp.zeros(self.nu))

        self._shift = shift

    # ---- 状态注入 ----
    def init_state(self, q, qd):
        """初始化 MPC state 为标准站姿（rollout 从站姿开始，稳定不 NaN）。"""
        st = self.env.reset(jax.random.PRNGKey(0))
        info = dict(st.info)
        info["step"] = 0
        self.state = st.replace(info=info)
        self.Y = jnp.zeros((self.H + 1, self.nu))

    def set_cmd(self, vx, vy, vyaw):
        self.cmd_vel = jnp.array([vx, vy, 0.0])
        self.cmd_ang = jnp.array([0.0, 0.0, vyaw])

    def _update_state(self, q, qd, t):
        """注入真实状态到 rollout 起点（qpos/qvel），并更新指令 info。

        旧方案从标准站姿 rollout（真实状态注入曾 NaN）；新模型（轮接触 + 躯干
        兜底 + 速度伺服 + 控制量裁剪）下真实状态 rollout 稳定，方向一致性
        大幅改善。若 rollout 发散，MPPI 侧 NaN 防御会把坏轨迹置为 -1e6。
        """
        info = dict(self.state.info)
        info["step"] = int(t / self.dt)
        info["vel_tar"] = self.cmd_vel
        info["ang_vel_tar"] = self.cmd_ang
        d = self.state.pipeline_state.data
        q = jnp.asarray(q, dtype=jnp.float32)
        qd = jnp.asarray(qd, dtype=jnp.float32)
        d = d.replace(qpos=q, qvel=qd)
        ps = self.state.pipeline_state.replace(data=d)
        self.state = self.state.replace(pipeline_state=ps, info=info)

    # ---- 单步规划（shift + MPPI）----
    def plan_once(self, q, qd, t):
        self._update_state(q, qd, t)
        self.Y = self._shift(self.Y)
        self.Y = self._mppi(self.state, self.Y, self.rng, self.temp, self.sigma)
        self.Y = self.Y.block_until_ready()
        return self.Y[0]

    def get_tau(self, action):
        tau = self.env.act2tau(action, self.state.pipeline_state)
        return np.asarray(tau)

    # ---- 异步规划线程 ----
    def update_plan_state(self, q, qd, t):
        """仿真线程每步调用：更新最新状态快照供规划线程读取。"""
        with self._lock:
            self._plan_q = np.asarray(q, dtype=np.float32)
            self._plan_qd = np.asarray(qd, dtype=np.float32)
            self._plan_t = float(t)

    def start_planning(self, q, qd):
        if self.state is None:
            self.init_state(q, qd)
        self._plan_q = np.asarray(q, dtype=np.float32)
        self._plan_qd = np.asarray(qd, dtype=np.float32)
        self._plan_t = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._plan_loop, daemon=True)
        self._thread.start()

    def _plan_loop(self):
        first = True
        while not self._stop.is_set():
            t0 = time.time()
            try:
                with self._lock:
                    q = self._plan_q.copy()
                    qd = self._plan_qd.copy()
                    t = self._plan_t
                act = self.plan_once(q, qd, t)
                tau = self.get_tau(act)
                if np.any(np.isnan(tau)) or np.any(np.isinf(tau)):
                    continue
                with self._lock:
                    self.latest_tau = tau
                if first:
                    print(f"[MPPI] 规划线程就绪（{time.time()-t0:.2f}s/次）", flush=True)
                    first = False
            except Exception as e:
                import traceback
                print(f"[MPPI] 规划线程异常: {e}", flush=True)
                traceback.print_exc()
                time.sleep(1.0)

    def stop_planning(self):
        self._stop.set()
