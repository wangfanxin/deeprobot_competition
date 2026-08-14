"""Evaluate a trained RL-stair checkpoint on curriculum stages (read-only).

Loads a PPO checkpoint and rolls out the noiseless actor on the existing
S10RLEnv (no env param/gate changes). Reports per-stage success / fall /
timeout / unfinished rates, mean forward progress and risers crossed.

Usage:
  python rl_stair/eval.py --ckpt rl_stair/logs/model_latest.pt \
      --stages T3_stairs6 T5_mixed T6_handoff \
      --num_envs 128 --episodes 5 --max_steps 500 --seed 1
"""
import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp
import torch

from rl_stair.configs.rl_stair_config import make_stages, PPOCfg
from rl_stair.ppo import PPO


def to_torch(x, device):
    return torch.as_tensor(np.asarray(x), device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--stages", type=str, default="T3_stairs6,T5_mixed,T6_handoff")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=0)   # 0 -> cfg.max_ep_len
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"checkpoint not found: {args.ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ckpt={args.ckpt} stages={args.stages} "
          f"num_envs={args.num_envs} episodes={args.episodes}")

    jax.config.update("jax_default_matmul_precision", "float32")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stages = make_stages(args.num_envs)
    want = [s.strip() for s in args.stages.split(",") if s.strip()]
    stages = [s for s in stages if s.name in want]
    if not stages:
        sys.exit(f"no matching stages for {want}; available="
                 + ",".join(s.name for s in make_stages(args.num_envs)))

    ppo_cfg = PPOCfg(num_envs=args.num_envs)
    for st in stages:
        env = None
        try:
            env = S10RLEnv(st.make_env_cfg())
        except NameError:
            from rl_stair.envs.s10_env import S10RLEnv
            env = S10RLEnv(st.make_env_cfg())
        t0 = time.time()
        ppo = PPO(env.obs_dim, env.priv_dim, env.action_size, ppo_cfg, device)
        ppo.load(args.ckpt)
        reset_j = jax.jit(env.reset)
        step_j = jax.jit(env.step)
        reset_state_j = jax.jit(env.reset_state)
        obs_of_j = jax.jit(env.obs_of)
        priv_of_j = jax.jit(env.priv_of)
        max_steps = args.max_steps or int(env.cfg.max_ep_len)
        print(f"[{st.name}] env built (obs={env.obs_dim}, priv={env.priv_dim}) "
              f"compile {time.time()-t0:.0f}s")

        agg = {"succ": 0.0, "fall": 0.0, "timeout": 0.0, "unfinished": 0.0,
               "prog": 0.0, "risers": 0.0, "rew": 0.0}
        for ep in range(args.episodes):
            state, obs, priv = reset_j(jax.random.PRNGKey(args.seed * 1000 + ep))
            n = env.n
            finished = np.zeros(n, dtype=bool)
            ep_succ = np.zeros(n, dtype=bool)
            ep_fall = np.zeros(n, dtype=bool)
            ep_timeout = np.zeros(n, dtype=bool)
            ep_prog = np.zeros(n)
            ep_ris = np.zeros(n, dtype=np.int32)
            ep_rew = np.zeros(n)
            step = 0
            while not bool(finished.all()) and step < max_steps:
                obs_t = to_torch(obs, device)
                with torch.no_grad():
                    a = ppo.actor.act(obs_t, noiseless=True)
                a_np = a.detach().cpu().numpy()
                state, obs, priv, rew, done, succ = step_j(state, jnp.asarray(a_np))
                done_np = np.asarray(done)
                succ_np = np.asarray(succ)
                ep_len_np = np.asarray(state["ep_len"])
                just = done_np & ~finished
                if bool(just.any()):
                    ep_succ |= just & succ_np
                    ep_timeout |= just & (~succ_np) & (ep_len_np >= int(env.cfg.max_ep_len))
                    ep_fall |= just & (~succ_np) & (ep_len_np < int(env.cfg.max_ep_len))
                    base_y = np.asarray(state["data"].qpos[:, 1])
                    prog = np.maximum(0.0, base_y - env.start_y)
                    risers = np.sum(np.asarray(env.riser_y)[None, :] < base_y[:, None], axis=-1)
                    ep_prog = np.where(just, prog, ep_prog)
                    ep_ris = np.where(just, risers, ep_ris)
                    finished |= just
                    # auto-reset done envs to keep the sim stable (same as train.py)
                    rs = reset_state_j(jax.random.fold_in(state["rng"], step))
                    state = env.merge_reset(state, done_np, rs)
                    obs = obs_of_j(state)
                    priv = priv_of_j(state, obs)
                ep_rew += np.asarray(rew)
                step += 1
            n_fin = max(int(finished.sum()), 1)
            agg["succ"] += float(ep_succ.mean())
            agg["fall"] += float(ep_fall.mean())
            agg["timeout"] += float(ep_timeout.mean())
            agg["unfinished"] += float((~finished).mean())
            agg["prog"] += float(ep_prog.sum() / n_fin)
            agg["risers"] += float(ep_ris.sum() / n_fin)
            agg["rew"] += float(ep_rew.mean())

        E = max(args.episodes, 1)
        print(f"  RESULT stage={st.name} ep={args.episodes} "
              f"succ={agg['succ']/E:.3f} fall={agg['fall']/E:.3f} "
              f"timeout={agg['timeout']/E:.3f} unfinished={agg['unfinished']/E:.3f} "
              f"prog={agg['prog']/E:.2f}m risers={agg['risers']/E:.1f} rew={agg['rew']/E:.2f}")


if __name__ == "__main__":
    main()
