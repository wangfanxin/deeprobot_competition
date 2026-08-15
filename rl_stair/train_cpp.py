"""train_cpp.py: train RL-stair directly in C++ MuJoCo (CPU) - USER-APPROVED 2026-08-15.

MJX<->C++ 2.7x wheel-drive gap makes MJX-trained policies unreliable in the official
C++ sim. This trains in C++ MuJoCo (S10 robot + box competition stairs) so the policy
learns real C++ contact dynamics. Starts from the best MJX policy (policy.pt) and
refines toward high official-env (S10_track.xml) success.

Reward (focused): dense progress toward wp7 + goal + upright + fall termination.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from rl_stair.envs.s10_env_cpp import S10EnvCPP, REACH_Y
from rl_stair.ppo import PPO, PPOCfg

ROLLOUT = 24
R_PROGRESS, R_GOAL, R_UPRIGHT = 4.0, 10.0, -1.0
SPAWN_Y = 32.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default="rl_stair/deploy/policy.pt")
    ap.add_argument("--logdir", default="rl_stair/logs_cpp")
    ap.add_argument("--official", action="store_true", help="train on official S10_track.xml (mesh)")
    args = ap.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    env = S10EnvCPP(num_envs=args.envs, official=args.official)
    ppo = PPO(env.obs_dim, env.obs_dim + 3 + 1 + 12 + 1, env.action_size,
              PPOCfg(num_envs=args.envs, num_steps_per_env=ROLLOUT, lr=args.lr), "cpu")
    ppo.init_buffer(args.envs, ROLLOUT, env.obs_dim, env.obs_dim + 3 + 1 + 12 + 1, env.action_size)
    if args.init and os.path.exists(args.init):
        # init actor/critic from the MJX best policy (load state dict if possible)
        try:
            ck = torch.load(args.init, map_location="cpu")
            if "actor" in ck:
                ppo.actor.load_state_dict(ck["actor"])
                ppo.critic.load_state_dict(ck["critic"])
                print("init from", args.init)
        except Exception as e:
            print("init skip:", e)

    obs = env.reset()
    best_reach = -1.0
    priv_pad = torch.zeros(args.envs, 17)
    for upd in range(args.updates):
        t0 = time.time()
        obs_np = obs
        # rollout
        for s_i in range(ROLLOUT):
            obs_t = torch.as_tensor(obs_np)
            with torch.no_grad():
                a, logp = ppo.act(obs_t)
            a_np = a.numpy()
            obs_next, rew, done, succ = env.step(a_np)
            # rewards: progress + goal + upright + fall
            pitch = np.array([abs(float(np.arcsin(np.clip(
                2*(env.data[k].qpos[4]*env.data[k].qpos[6] - env.data[k].qpos[5]*env.data[k].qpos[3]),
                -1, 1)))) for k in range(env.n)])
            rew = rew.copy()
            rew += R_PROGRESS * 0.005 * np.maximum(0.0, np.array([env.data[k].qpos[1] for k in range(env.n)]) - SPAWN_Y)
            rew += R_UPRIGHT * 0.005 * (pitch ** 2)
            rew += R_GOAL * succ
            rew -= 1.0 * done
            priv = torch.cat([obs_t, priv_pad], dim=1)
            with torch.no_grad():
                v = ppo.critic(priv)
            ppo.store(s_i, obs_t, priv, a, logp,
                      torch.as_tensor(rew, dtype=torch.float32), torch.as_tensor(done), v)
            # reset done envs
            if done.any():
                for k in np.where(done)[0]:
                    rng = np.random.default_rng(5000 + upd * 100 + k)
                    d = env.data[k]
                    x = float(rng.uniform(-0.1, 0.1)); yaw = 1.5708 + float(rng.uniform(-0.05, 0.05))
                    d.qpos[:] = 0; d.qvel[:] = 0
                    d.qpos[0:3] = [x, SPAWN_Y, 0.9]
                    d.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                    for i, j in enumerate(env.idx["act2jnt"]): d.qpos[j] = env.default_dof[i]
                    import mujoco as _mj
                    _mj.mj_forward(env.model, d)
                    env.ep_len[k] = 0
            obs = obs_next
        with torch.no_grad():
            last_v = ppo.critic(torch.cat([torch.as_tensor(obs), priv_pad], dim=1))
        stats = ppo.update(last_v)
        sr = float(np.mean(succ))
        if sr > best_reach:
            best_reach = sr
            torch.save({"actor": ppo.actor.state_dict(), "critic": ppo.critic.state_dict(),
                        "it": upd}, os.path.join(args.logdir, "model_best_cpp.pt"))
        if upd % 50 == 0:
            print(f"upd={upd} reach={sr:.3f} best={best_reach:.3f} "
                  f"std={stats['mean_std']:.3f} loss_a={stats['loss_actor']:.3f} dt={time.time()-t0:.1f}s", flush=True)
    torch.save({"actor": ppo.actor.state_dict(), "critic": ppo.critic.state_dict()},
               os.path.join(args.logdir, "model_final_cpp.pt"))


def mujoco_forward(env, k):
    import mujoco
    mujoco.mj_forward(env.model, env.data[k])


if __name__ == "__main__":
    main()
