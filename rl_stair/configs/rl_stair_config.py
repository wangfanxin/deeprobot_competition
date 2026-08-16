"""RL-stair training configs: env, PPO, curriculum stages."""
from dataclasses import dataclass, field
from rl_stair.envs.s10_env import S10RLCfg
from rl_stair.envs.terrain import flat, single_step, stairs

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
# T5/T6 mixed uses a SHORTER staircase (0.061+0.125x3): full 6-step sequence + 2 ridges
# destabilizes the policy after the 2nd ridge (T5 bounced 12+ times, 11:20). 4-step keeps
# the mixed-sequence skill; full 6-step mastered in T3.

def _base_cfg(num_envs=1024):
    c = S10RLCfg(num_envs=num_envs, seed=0)
    c.max_ep_len = 500
    # approach-angle randomization 09:55 (user): +-0.7rad (~+-40deg) around track heading;
    # T0 keeps +-0.1 (gentle warm-up). terrain_ctx now yaw-rotates offsets to match.
    c.yaw_lo, c.yaw_hi = -0.7, 0.7
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
        # USER-DIRECTED 2026-08-15: T2 split into ROW of low steps, raise height gradually.
        c = _base_cfg(num_envs); c.terrain = stairs([0.05]*4, y0=1.5); return c
    def c2b():
        c = _base_cfg(num_envs); c.terrain = stairs([0.061]*4, y0=1.5); return c
    def c2c():
        # 4x8cm added 02:35: 6.1->10 jump too steep (T2c 4x10 failed 3x at succ 0.000)
        c = _base_cfg(num_envs); c.terrain = stairs([0.08]*4, y0=1.5); return c
    def c2d():
        # close-spawn (0.3-0.8m before step): isolate the >8cm lift-climb skill (06:50)
        c = _base_cfg(num_envs); c.terrain = stairs([0.10]*4, y0=1.5)
        c.spawn_back_lo, c.spawn_back_hi = 0.3, 0.8
        return c
    def c2e():
        c = _base_cfg(num_envs); c.terrain = stairs([0.125]*4, y0=1.5)
        c.spawn_back_lo, c.spawn_back_hi = 0.3, 0.8
        return c
    def c3():
        c = _base_cfg(num_envs); c.terrain = stairs(COMPETITION_RISERS, y0=1.5)
        c.spawn_back_lo, c.spawn_back_hi = 0.3, 0.8
        return c
    # USER-DIRECTED 2026-08-15 21:50: ALL RIDGES REMOVED (T4a/b/c, T5 mixed).
    # RL-stair covers STAIRS ONLY; sim2sim is the acceptance bar.
    # T6 = competition 6-step staircase + worst-case handoff approach (yaw +-1.0, vx -0.5..2.5).
    def c6():
        c = _base_cfg(num_envs)
        c.terrain = stairs(COMPETITION_RISERS, y0=1.5)
        c.max_ep_len = 1000
        # USER-DIRECTED 2026-08-16: realistic handoff distribution. The previous yaw
        # +-1.0 rad REGRESSED the aligned case (official real-mesh 100% -> 53%) - the
        # cruise handoff yaw is only +-0.2 rad, so keep +-0.3 margin. Add initial-pose
        # DR (squat_frac + leg_q_jit) - verified: squat-start = 0/30, leg_jit 0.3 = 70%
        # on the real mesh without it; the handoff delivers legs with up to ~0.15 rad
        # error (and possibly squat if the transition is cut short).
        c.yaw_lo, c.yaw_hi = -0.3, 0.3
        c.squat_frac = 0.35
        c.leg_q_jit = 0.25
        return c
    return [
        Stage("T0_flat", c0, advance_at=0.5, min_iters=30),
        Stage("T1a_step005", c1a, advance_at=0.4, min_iters=50),
        Stage("T1b_step081", c1b, advance_at=0.4, min_iters=50),
        Stage("T1c_step010", c1c, advance_at=0.4, min_iters=50),
        # T1d(single 12.5cm) REMOVED 2026-08-15 01:50: single 12.5cm gate too tight
        # (2.2cm lift margin) -> 6+ regresses. Row-of-steps T2a-d reaches 12.5cm via
        # height gradient (user-guided); T1c(10cm single) -> T2a(4x5cm row) -> ... -> T3.
        Stage("T2a_stairs4x005", c2a, advance_at=0.4, min_iters=50),
        Stage("T2b_stairs4x0061", c2b, advance_at=0.35, min_iters=80),
        Stage("T2c_stairs4x008", c2c, advance_at=0.35, min_iters=80),
        Stage("T2d_stairs4x010", c2d, advance_at=0.3, min_iters=80),
        Stage("T2e_stairs4x0125", c2e, advance_at=0.6, min_iters=150),   # gate raised 12:20 (transfer margin)
        Stage("T3_stairs6", c3, advance_at=0.6, min_iters=300),   # gate raised 12:20 (sim2sim: 0.30->1/6 transfer fail)
        # T4a/b/c ridges + T5 mixed REMOVED (USER-DIRECTED 2026-08-15 21:50: no ridges)
        # BUGFIX 2026-08-15 21:55: regress_below=0.0 - terminal stage must NOT bounce
        # back to T3 (min_iters=0 + default regress_below 0.05 made it regress at once).
        Stage("T6_handoff", c6, advance_at=0.0, min_iters=0, max_iters=999999, regress_below=0.0),
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
