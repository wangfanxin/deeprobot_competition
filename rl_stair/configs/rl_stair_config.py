"""RL-stair training configs: env, PPO, curriculum stages."""
from dataclasses import dataclass, field
from rl_stair.envs.s10_env import S10RLCfg
from rl_stair.envs.terrain import flat, single_step, stairs, ridge, mixed

# ---------------------------------------------------------------------------
# Curriculum stages (T0..T6). Each stage = env cfg override + success gate.
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    name: str
    make_env_cfg: callable          # () -> S10RLCfg
    success_metric: str = "succ"    # which info metric gates advancement
    advance_at: float = 0.35        # fraction of envs succeeding -> advance
    regress_below: float = 0.05     # fraction -> regress
    min_iters: int = 50             # min iterations before advancing
    max_iters: int = 3000

COMPETITION_RISERS = [0.061, 0.125, 0.125, 0.125, 0.125, 0.125]

def _base_cfg(num_envs=1024):
    c = S10RLCfg(num_envs=num_envs, seed=0)
    c.max_ep_len = 500
    return c

def make_stages(num_envs=1024):
    def c0():
        # T0 warm-up gate lowered 6m->4.5m (2026-08-14 22:12, doc §3.12 fallback):
        #   converged policy (std 0.38) sustains ~0.5 m/s -> reaches y~5.9 but only ~5-7%
        #   cross 6.0 in 16s from spawn y[-2,0] (marginal 6-8m course). 4.5m passes at
        #   0.28 m/s; REAL speed (0.8-1.8, incl 1.5 entry) is taught in T1+.
        c = _base_cfg(num_envs); c.terrain = flat(course=4.5)
        # gentle start for WIP balancing: small initial speed / pose / vel perturbation
        c.vx_lo, c.vx_hi = 0.0, 0.5
        c.yaw_lo, c.yaw_hi = -0.1, 0.1
        c.q_off = 0.02
        c.v_off = 0.2
        c.cmd_vx_lo, c.cmd_vx_hi = 0.4, 0.9   # T0 target speed reachable -> succ achievable
        c.max_ep_len = 800   # 16s
        return c
    def c1a():
        c = _base_cfg(num_envs); c.terrain = single_step(0.05, y0=1.5); return c
    def c1b():
        c = _base_cfg(num_envs); c.terrain = single_step(0.081, y0=1.5); return c
    def c1c():
        c = _base_cfg(num_envs); c.terrain = single_step(0.10, y0=1.5); return c
    def c1d():
        c = _base_cfg(num_envs); c.terrain = single_step(0.125, y0=1.5); return c
    def c2a():
        # USER-DIRECTED 2026-08-15 01:40: T2 split into a ROW of low steps, then raise
        # height gradually. Learn multi-step sequencing rhythm at easy heights first.
        c = _base_cfg(num_envs); c.terrain = stairs([0.05]*4, y0=1.5); return c
    def c2b():
        c = _base_cfg(num_envs); c.terrain = stairs([0.061]*4, y0=1.5); return c
    def c2c():
        c = _base_cfg(num_envs); c.terrain = stairs([0.10]*4, y0=1.5); return c
    def c2d():
        c = _base_cfg(num_envs); c.terrain = stairs([0.125]*4, y0=1.5); return c
    def c3():
        c = _base_cfg(num_envs); c.terrain = stairs(COMPETITION_RISERS, y0=1.5); return c
    def c4a():
        c = _base_cfg(num_envs); c.terrain = ridge(0.08, y0=1.5); return c
    def c4b():
        c = _base_cfg(num_envs); c.terrain = ridge(0.12, y0=1.5); return c
    def c4c():
        c = _base_cfg(num_envs); c.terrain = ridge(0.15, y0=1.5); return c
    def c5():
        c = _base_cfg(num_envs)
        c.terrain = mixed([("ridge", dict(height=0.12, y0=1.5)),
                           ("stairs", dict(risers=COMPETITION_RISERS, y0=4.5)),
                           ("ridge", dict(height=0.15, y0=12.0))])
        return c
    def c6():
        c = _base_cfg(num_envs)
        c.terrain = mixed([("ridge", dict(height=0.12, y0=1.5)),
                           ("stairs", dict(risers=COMPETITION_RISERS, y0=4.5)),
                           ("ridge", dict(height=0.15, y0=12.0))])
        return c
    return [
        Stage("T0_flat", c0, advance_at=0.5, min_iters=30),
        Stage("T1a_step005", c1a, advance_at=0.4, min_iters=50),
        Stage("T1b_step081", c1b, advance_at=0.4, min_iters=50),
        Stage("T1c_step010", c1c, advance_at=0.4, min_iters=50),
        Stage("T1d_step0125", c1d, advance_at=0.35, min_iters=80),
        Stage("T2a_stairs4x005", c2a, advance_at=0.4, min_iters=50),
        Stage("T2b_stairs4x0061", c2b, advance_at=0.35, min_iters=80),
        Stage("T2c_stairs4x010", c2c, advance_at=0.3, min_iters=80),
        Stage("T2d_stairs4x0125", c2d, advance_at=0.3, min_iters=100),
        Stage("T3_stairs6", c3, advance_at=0.3, min_iters=200),
        Stage("T4a_ridge008", c4a, advance_at=0.5, min_iters=50),
        Stage("T4b_ridge012", c4b, advance_at=0.4, min_iters=80),
        Stage("T4c_ridge015", c4c, advance_at=0.35, min_iters=80),
        Stage("T5_mixed", c5, advance_at=0.25, min_iters=200),
        Stage("T6_handoff", c6, advance_at=0.0, min_iters=0, max_iters=999999),
    ]

# ---------------------------------------------------------------------------
# PPO config
# ---------------------------------------------------------------------------
@dataclass
class PPOCfg:
    num_envs: int = 1024
    num_steps_per_env: int = 24        # rollout length (policy steps)
    num_minibatches: int = 4
    num_epochs: int = 5
    lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.003
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    actor_units: tuple = (256, 256, 128)
    critic_units: tuple = (256, 256, 128)
    init_noise_std: float = 1.0
    lr_schedule: str = "linear"        # "linear" | "constant"
    save_every: int = 200              # iterations
    log_every: int = 10
    resume: str = ""
