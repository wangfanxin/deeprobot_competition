import os
import time
from dataclasses import dataclass
import importlib
import sys

import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
import art
import emoji

import jax
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import functools

from brax.io import html
import brax.envs as brax_envs

import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig

plt.style.use("science")

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states


@jax.jit
def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


CE_ELITE = float(os.environ.get("S10_CE_ELITE", "0.0"))
CE_ALPHA = float(os.environ.get("S10_CE_ALPHA", "0.5"))


class MBDPI:
    def __init__(self, args: DialConfig, env, ctx=None):
        # v182：左右腿采样噪声对称化（fl/fr、hl/hr 共享噪声，动作天然对称）。
        # 爬梯侧翻主因之一是采样不对称（左/右轮独立噪声 → 一高一低）。
        # 仅腿维（0:12）；轮维保留差速（转向需要）。env 开关，静态不 retrace。
        self.sym_sample = bool(int(os.environ.get("S10_SYM_SAMPLE", "0")))
        # v192 CE 精英混合（模块级常量，JIT 静态分支）
        CE_ELITE = float(os.environ.get("S10_CE_ELITE", "0.0"))
        CE_ALPHA = float(os.environ.get("S10_CE_ALPHA", "0.5"))
        self.args = args
        self.env = env
        # 预热缓存重构：ctx 模式（S10）把 rollout 参数化，JAX persistent
        # cache 才能命中（闭包捕获 env 的对象哈希跨进程不稳定，永远 miss）。
        self._ctx = ctx
        self.nu = env.action_size
        # per-action-dim 采样噪声缩放（None=各向同性）：定向增大腿维探索，
        # 提高"采样搜到抬腿轨迹"的概率（0804 §3.5，爬坡成功率实验）。
        if getattr(args, "sigma_dim", None):
            self.sigma_dim = jnp.asarray(args.sigma_dim, dtype=jnp.float32)
            if self.sigma_dim.shape[0] != self.nu:
                raise ValueError(
                    f"sigma_dim 长度 {self.sigma_dim.shape[0]} != nu {self.nu}")
        else:
            self.sigma_dim = jnp.ones(self.nu)

        self.update_fn = {
            "mppi": softmax_update,
        }[args.update_method]

        sigma0 = 1e-2
        sigma1 = 1.0
        sigma_scale = args.sigma_scale
        A = sigma0
        B = jnp.log(sigma1 / sigma0) / args.Ndiffuse
        self.sigma_control = (
            args.horizon_diffuse_factor ** jnp.arange(args.Hnode + 1)[::-1]
        )

        self.sigma_control *= sigma_scale
        print(self.sigma_control)

        # node to u
        self.ctrl_dt = 0.02
        self.step_us = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_nodes = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

        # setup function
        if self._ctx is not None:
            from dial_mpc.envs.s10_env import s10_rollout_us
            self.rollout_us = jax.jit(s10_rollout_us)
        else:
            rollout_step = getattr(
                self.env, "step_rollout", None) or self.env.step
            self.rollout_us = jax.jit(
                functools.partial(rollout_us, rollout_step))
        self.rollout_us_vmap = jax.jit(
            jax.vmap(self.rollout_us, in_axes=(None, None, 0)))
        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
        )  # process (horizon, node)
        self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=(0))
        )  # process (batch, horizon, node)
        self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
        us = spline(self.step_us)
        return us

    @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
        nodes = spline(self.step_nodes)
        return nodes

    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale, sigma_dim):
        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_Y = jax.random.normal(
            Y0s_rng, (self.args.Nsample, self.args.Hnode + 1, self.nu)
        )
        if self.sym_sample:
            # 左右腿共享噪声：fl/fr (0:3/3:6)、hl/hr (6:9/9:12) 取均值
            _m_lf = (eps_Y[:, :, 0:3] + eps_Y[:, :, 3:6]) / 2.0
            eps_Y = eps_Y.at[:, :, 0:3].set(_m_lf)
            eps_Y = eps_Y.at[:, :, 3:6].set(_m_lf)
            _m_lr = (eps_Y[:, :, 6:9] + eps_Y[:, :, 9:12]) / 2.0
            eps_Y = eps_Y.at[:, :, 6:9].set(_m_lr)
            eps_Y = eps_Y.at[:, :, 9:12].set(_m_lr)
            # 采样均值也对称化（否则样本绕非对称均值，MPPI 平均仍不对称）
            Ybar_i = Ybar_i.at[:, 0:3].set(
                (Ybar_i[:, 0:3] + Ybar_i[:, 3:6]) / 2.0)
            Ybar_i = Ybar_i.at[:, 3:6].set(Ybar_i[:, 0:3])
            Ybar_i = Ybar_i.at[:, 6:9].set(
                (Ybar_i[:, 6:9] + Ybar_i[:, 9:12]) / 2.0)
            Ybar_i = Ybar_i.at[:, 9:12].set(Ybar_i[:, 6:9])
        Y0s = eps_Y * noise_scale[None, :, None] * sigma_dim[None, None, :] \
            + Ybar_i
        # we can't change the first control
        Y0s = Y0s.at[:, 0].set(Ybar_i[0, :])
        # append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = jnp.concatenate([Y0s, Ybar_i[None]], axis=0)
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        # convert Y0s to us
        us = self.node2u_vvmap(Y0s)
        # spline interpolation can overshoot the [-1,1] action bounds; re-clip so
        # rollout actions never exceed the actuator range (prevents NaN collapse)
        us = jnp.clip(us, -1.0, 1.0)

        # esitimate mu_0tm1
        if self._ctx is not None:
            rewss, pipeline_statess = self.rollout_us_vmap(self._ctx, state, us)
        else:
            rewss, pipeline_statess = self.rollout_us_vmap(state, us)
        # defensive: one divergent trajectory must not poison the softmax update
        rewss = jnp.where(jnp.isnan(rewss) | jnp.isinf(rewss), -1e6, rewss)
        # 发散样本可能是"巨大但有穷"的奖励（实测 -1e32，来自速度爆炸 1e15），
        # NaN 过滤拦不住，会毒化 rews.std() 使 softmax 退化为均匀权重、
        # MPPI 无法移动解。这里统一夹到 [-1e4, 1e4]（正常步奖励 ~-3e3）。
        rewss = jnp.clip(rewss, -1e4, 1e4)
        rew_Ybar_i = rewss[-1].mean()
        qss = pipeline_statess.q
        qdss = pipeline_statess.qd
        xss = pipeline_statess.x.pos
        rews = rewss.mean(axis=-1)
        logp0 = (rews - rew_Ybar_i) / (rews.std(axis=-1) + 1e-6) / self.args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)

        # NOTE: update only with reward
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)
        # v192 CE 精英混合（MPOPI 文献 Keshavarz 2025）：softmax 均值被
        # 大量平庸样本拖拽，弯道（S 弯）采样方差大。取 top-K 精英轨迹
        # 均值按比例混合，压低坏样本污染。K=Nsample*S10_CE_ELITE，
        # 混合比 S10_CE_ALPHA（0=关，纯 MPPI）。
        if CE_ELITE > 0.0:
            _K = max(2, int(self.args.Nsample * CE_ELITE))
            _elite = jnp.take(Y0s, jnp.argsort(-rews)[:_K], axis=0)
            Ybar = (1.0 - CE_ALPHA) * Ybar + CE_ALPHA * jnp.mean(_elite, axis=0)
        qbar = jnp.einsum("n,nij->ij", weights, qss)
        qdbar = jnp.einsum("n,nij->ij", weights, qdss)
        xbar = jnp.einsum("n,nijk->ijk", weights, xss)

        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": new_noise_scale,
            # v209: 样本动作（含 Ybar 追加行，末维）。供 MPOPI 式精英协方差
            # 自适应（host 侧选 top-K 算每维 std/mean，更新下一轮 sigma_dim
            # 与精英均值偏置）——数据驱动"学习好样本分布"。
            "Y0s": Y0s,
        }

        return rng, Ybar, info

    def reverse(self, state, YN, rng):
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                rng, Yi, rews = self.reverse_once(
                    state, rng, Yi,
                    self.sigmas[i] * jnp.ones(self.args.Hnode + 1),
                    self.sigma_dim,
                )
                Yi.block_until_ready()
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{rews.mean():.2e}", "freq": f"{freq:.2f}"})
        return Yi

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        Y = self.u2node_vmap(u)
        return Y

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift_n(self, Y, n):
        """按真实 plan 间隔推进 n 个 ctrl_dt（方案 C，2026-08-07）：
        执行零阶保持周期 = plan 周期时，shift 必须推进相同步数，
        否则 Y 序列相位每轮错位。n 为动态 jnp 标量，不触发 retrace。"""
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -n, axis=0)
        idx = jnp.arange(u.shape[0], dtype=jnp.int32)
        n_i = jnp.asarray(n, dtype=jnp.int32)
        zero_mask = idx >= (u.shape[0] - n_i)
        u = jnp.where(zero_mask[:, None], 0.0, u)
        Y = self.u2node_vmap(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        u = jnp.roll(u, -n_step, axis=0)
        u = u.at[-n_step:].set(jnp.zeros_like(u[-n_step:]))
        Y = self.u2node_vmap(u)
        return Y


def main():

    def reverse_scan(rng_Y0_state, factor):
        rng, Y0, state = rng_Y0_state
        rng, Y0, info = mbdpi.reverse_once(state, rng, Y0, factor)
        return (rng, Y0, state), info

    art.tprint("LeCAR @ CMU\nDIAL-MPC", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    config_or_example.add_argument("--list-examples", action="store_true")
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    args = parser.parse_args()

    if args.list_examples:
        print("Examples:")
        for example in examples:
            print(f"  {example}")
        return

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    rng = jax.random.PRNGKey(seed=dial_config.seed)

    # find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    mbdpi = MBDPI(dial_config, env)

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    YN = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])

    rng_exp, rng = jax.random.split(rng)
    # Y0 = mbdpi.reverse(state_init, YN, rng_exp)
    Y0 = YN

    Nstep = dial_config.n_steps
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []
    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # forward single step
            state = step_env(state, Y0[0])
            rollout.append(state.pipeline_state)
            rews.append(state.reward)
            us.append(Y0[0])

            # update Y0
            Y0 = mbdpi.shift(Y0)

            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
                print("Performing JIT on DIAL-MPC")

            t0 = time.time()
            traj_diffuse_factors = (
                mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
            )
            (rng, Y0, _), info = jax.lax.scan(
                reverse_scan, (rng, Y0, state), traj_diffuse_factors
            )
            rews_plan.append(info["rews"][-1].mean())
            infos.append(info)
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # save us
    # us = jnp.array(us)
    # jnp.save("./results/us.npy", us)

    # create result dir if not exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # plot rews_plan
    # plt.plot(rews_plan)
    # plt.savefig(os.path.join(dial_config.output_dir,
    #             f"{timestamp}_rews_plan.pdf"))

    # host webpage with flask
    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
    )

    # save the html file
    with open(
        os.path.join(dial_config.output_dir, f"{timestamp}_brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    # save the rollout
    data = []
    xdata = []
    for i in range(len(rollout)):
        pipeline_state = rollout[i]
        data.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    pipeline_state.qpos,
                    pipeline_state.qvel,
                    pipeline_state.ctrl,
                ]
            )
        )
        xdata.append(infos[i]["xbar"][-1])
    data = jnp.array(data)
    xdata = jnp.array(xdata)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), data)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_predictions"), xdata)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main()
