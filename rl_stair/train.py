"""RL-stair training loop with auto-curriculum + resource monitoring.

Usage:
  python rl_stair/train.py --num_envs 1024 --max_iters 3000 [--resume PATH]
"""
import os, sys, time, argparse, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import torch

from rl_stair.configs.rl_stair_config import make_stages, PPOCfg
from rl_stair.envs.s10_env import S10RLEnv
from rl_stair.ppo import PPO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_torch(x, device):
    return torch.as_tensor(np.asarray(x), device=device)


def log_resource(path):
    try:
        out = []
        df = subprocess.run(["df", "-h", "/", "/mnt/c"], capture_output=True, text=True).stdout
        out.append(df.strip().replace("\n", " | "))
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        out.append("gpu=" + gpu)
        mem = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout.splitlines()[1]
        out.append("mem=" + mem)
        with open(path, "a") as f:
            f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] " + " ; ".join(out) + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=1024)
    ap.add_argument("--max_iters", type=int, default=3000)
    ap.add_argument("--stage", type=str, default="")       # force a single stage
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logdir", type=str, default=os.path.join(REPO, "rl_stair/logs"))
    ap.add_argument("--steps_per_env", type=int, default=24)
    ap.add_argument("--num_minibatches", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")        # quick test (few iters, no curriculum)
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open(os.path.join(args.logdir, "train.log"), "a", buffering=1)
    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    jax.config.update("jax_default_matmul_precision", "float32")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stages = make_stages(args.num_envs)
    if args.stage:
        stages = [s for s in stages if s.name == args.stage] or [stages[0]]

    ppo_cfg = PPOCfg(num_envs=args.num_envs, num_steps_per_env=args.steps_per_env,
                     num_minibatches=args.num_minibatches, num_epochs=args.epochs)

    # curriculum state
    stage_idx = 0
    env = None
    ppo = None
    state = None
    obs = None
    priv = None
    env_iters = 0
    succ_window = []

    def build_env(stage):
        nonlocal env, ppo, state, obs, priv
        env = S10RLEnv(stage.make_env_cfg())
        log(f"env built: {stage.name} stand_z={env.stand_z:.3f} obs={env.obs_dim}")
        ppo = PPO(env.obs_dim, env.priv_dim, env.action_size, ppo_cfg, device)
        if args.resume and os.path.exists(args.resume):
            ppo.load(args.resume)
            log(f"resumed from {args.resume} (iter {ppo.it})")
        ppo.init_buffer(args.num_envs, ppo_cfg.num_steps_per_env,
                        env.obs_dim, env.priv_dim, env.action_size)
        reset_j = jax.jit(env.reset)
        step_j = jax.jit(env.step)
        reset_state_j = jax.jit(env.reset_state)
        obs_of_j = jax.jit(env.obs_of)
        priv_of_j = jax.jit(env.priv_of)
        env.reset_j, env.step_j = reset_j, step_j
        env.reset_state_j, env.obs_of_j, env.priv_of_j = reset_state_j, obs_of_j, priv_of_j
        t0 = time.time()
        state, obs, priv = reset_j(jax.random.PRNGKey(args.seed + stage_idx))
        log(f"stage reset compiled {time.time()-t0:.0f}s")

    build_env(stages[0])
    obs_noise = torch.zeros(env.obs_dim, device=device)
    obs_noise[0:3] = 0.05
    obs_noise[3:6] = 0.01
    obs_noise[9:21] = 0.01
    obs_noise[21:33] = 0.05

    tot_env_int = 0
    it = 0
    t_start = time.time()
    log("=== training start ===")
    while it < args.max_iters:
        t_iter = time.time()
        # rollout
        ppo.optim.zero_grad()
        n_done_ep = 0
        n_succ_ep = 0
        for s_i in range(ppo_cfg.num_steps_per_env):
            obs_t = to_torch(obs, device)
            obs_t = obs_t + (2*torch.rand_like(obs_t) - 1) * obs_noise
            priv_t = to_torch(priv, device)
            with torch.no_grad():
                a, logp = ppo.act(obs_t)
                v = ppo.critic(priv_t)
            a_np = a.detach().cpu().numpy()
            t0 = time.time()
            state, obs, priv, rew, done, succ = env.step_j(state, jnp.asarray(a_np))
            # BUGFIX 2026-08-14: succ in log used last-rollout-step instantaneous flags only
            # (envs reset on done -> successes mid-episode invisible -> gate never advances).
            # Count done-events: every episode ends via success/fall/timeout; true rate =
            # succ_dones / total_dones, aggregated over the rollout.
            _d = np.asarray(done); _s = np.asarray(succ)
            n_done_ep += int(_d.sum())
            n_succ_ep += int((_d & _s).sum())
            dts = time.time() - t0
            if dts > 5.0:
                log(f"WARN slow step {dts:.1f}s at s_i={s_i}")
            # reset done envs immediately (avoids solver pathology on fallen robots)
            if bool(np.asarray(done).any()):
                reset_state = env.reset_state_j(jax.random.fold_in(state["rng"], env_iters * 1000 + s_i))
                state = env.merge_reset(state, done, reset_state)
                obs = env.obs_of_j(state)
                priv = env.priv_of_j(state, obs)
            ppo.store(s_i, obs_t.detach(), priv_t.detach(), a.detach(), logp.detach(),
                      to_torch(rew, device), to_torch(done, device), v.detach())
        # last value
        with torch.no_grad():
            last_v = ppo.critic(to_torch(priv, device))
        t_update = time.time()
        stats = ppo.update(last_v)
        iter_dt = time.time() - t_iter
        t_update = time.time() - t_update
        it += 1
        env_iters += 1
        tot_env_int += args.num_envs * ppo_cfg.num_steps_per_env

        succ_window.append(n_succ_ep / max(n_done_ep, 1))
        if len(succ_window) > 100:
            succ_window.pop(0)
        succ_rate = float(np.mean(succ_window)) if succ_window else 0.0
        iters_per_h = 3600.0 / max(time.time() - t_iter, 1e-6)
        el = time.time() - t_start
        eta = (args.max_iters - it) / max(iters_per_h, 1e-6)
        if it % ppo_cfg.log_every == 0:
            _q = np.asarray(state["data"].qpos)
            _by = _q[:, 1]; _qw,_qx,_qy2,_qz = _q[:,3],_q[:,4],_q[:,5],_q[:,6]
            _yaw = np.arctan2(2*(_qw*_qz+_qx*_qy2), 1-2*(_qy2*_qy2+_qz*_qz))
            _vl = np.asarray(state["data"].qvel)[:, 0:3]
            _tx = 2.0*(-_qy2*_vl[:,2] + _qz*_vl[:,1]); _ty = 2.0*(-_qz*_vl[:,0] + _qx*_vl[:,2])
            _tz = 2.0*(-_qx*_vl[:,1] + _qy2*_vl[:,0])
            _vbx = _vl[:,0] + _qw*_tx + (-_qy2*_tz + _qz*_ty)
            _wz = np.abs(np.asarray(state["data"].qvel)[:, 5])
            log(f"iter={it} stage={stages[stage_idx].name} succ={succ_rate:.3f} done={n_done_ep}/{n_succ_ep} "
                f"y_avg={float(_by.mean()):.2f} y_max={float(_by.max()):.2f} yaw={float(np.degrees(np.abs(_yaw).mean())):.0f} vbx={float(np.abs(_vbx).mean()):.2f} wz={float(_wz.mean()):.2f} "
                f"rew={float(np.mean(np.asarray(rew))):.2f} env_int={tot_env_int/1e6:.1f}M "
                f"steps/s={iters_per_h:.0f}/h eta={eta:.1f}h iter_dt={iter_dt:.1f}s upd={t_update:.1f}s "
                f"std={stats['mean_std']:.3f} loss_a={stats['loss_actor']:.4f}")
            log_resource(os.path.join(args.logdir, "resource.log"))
        if it % ppo_cfg.save_every == 0:
            ck = os.path.join(args.logdir, f"model_{it:06d}.pt")
            ppo.save(ck)
            latest = os.path.join(args.logdir, "model_latest.pt")
            ppo.save(latest)
            log(f"saved {ck}")
        if args.smoke and it >= 20:
            log("smoke done")
            break

        # curriculum advance/regress
        st = stages[stage_idx]
        if env_iters >= st.min_iters and it >= st.min_iters:
            if succ_rate >= st.advance_at and stage_idx < len(stages) - 1:
                log(f"ADVANCE {st.name} -> {stages[stage_idx+1].name} (succ={succ_rate:.3f})")
                stage_idx += 1
                env_iters = 0
                succ_window = []
                build_env(stages[stage_idx])
            elif succ_rate < st.regress_below and stage_idx > 0:
                log(f"REGRESS {st.name} -> {stages[stage_idx-1].name} (succ={succ_rate:.3f})")
                stage_idx -= 1
                env_iters = 0
                succ_window = []
                build_env(stages[stage_idx])
        if env_iters >= st.max_iters and stage_idx < len(stages) - 1:
            log(f"TIMEOUT ADVANCE {st.name} -> {stages[stage_idx+1].name}")
            stage_idx += 1
            env_iters = 0
            succ_window = []
            build_env(stages[stage_idx])

    ppo.save(os.path.join(args.logdir, "model_final.pt"))
    log(f"training finished: {it} iters, {tot_env_int/1e6:.1f}M env-int, {el/3600:.2f}h")
    logf.close()


if __name__ == "__main__":
    main()
