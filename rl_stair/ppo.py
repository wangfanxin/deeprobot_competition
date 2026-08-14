"""PPO (rsl_rl style, asymmetric actor-critic) for RL-stair."""
import os, time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from rl_stair.configs.rl_stair_config import PPOCfg


def mlp(in_dim, units, out_dim=None, act=nn.Tanh):
    layers = []
    d = in_dim
    for u in units:
        layers += [nn.Linear(d, u), act()]
        d = u
    if out_dim is not None:
        layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, units=(256, 256, 128), init_noise=1.0):
        super().__init__()
        self.mean = mlp(obs_dim, units, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(init_noise))))
        self.std = torch.exp(self.log_std)

    def forward(self, obs):
        return torch.tanh(self.mean(obs))

    def act(self, obs, noiseless=False):
        mu = self.forward(obs)
        self.std = torch.exp(self.log_std)
        if noiseless:
            return mu
        return (mu + torch.randn_like(mu) * self.std).clamp(-1.0, 1.0)

    def get_dist(self, obs):
        mu = self.forward(obs)
        self.std = torch.exp(self.log_std)
        return mu, self.std


class Critic(nn.Module):
    def __init__(self, priv_dim, units=(256, 256, 128)):
        super().__init__()
        self.net = mlp(priv_dim, units, 1)

    def forward(self, priv):
        return self.net(priv).squeeze(-1)


class RolloutBuffer:
    def __init__(self, num_envs, num_steps, obs_dim, priv_dim, action_dim, device):
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.priv = torch.zeros(num_steps, num_envs, priv_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)

    def compute_returns(self, last_value, gamma, lam):
        gae = 0.0
        for t in reversed(range(self.rewards.shape[0])):
            if t == self.rewards.shape[0] - 1:
                next_val = last_value
            else:
                next_val = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]


class PPO:
    def __init__(self, obs_dim, priv_dim, action_dim, cfg: PPOCfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.actor = Actor(obs_dim, action_dim, cfg.actor_units, cfg.init_noise_std).to(device)
        self.critic = Critic(priv_dim, cfg.critic_units).to(device)
        self.optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr, eps=1e-5)
        self.buffer = None
        self.it = 0

    def init_buffer(self, num_envs, num_steps, obs_dim, priv_dim, action_dim):
        self.buffer = RolloutBuffer(num_envs, num_steps, obs_dim, priv_dim, action_dim, self.device)
        self.buffer_idx = 0

    def _lr(self):
        if self.cfg.lr_schedule == "constant":
            return self.cfg.lr
        p = 1.0 - min(self.it / 5000.0, 1.0)
        return self.cfg.lr * max(p, 0.1)

    def act(self, obs_t):
        with torch.no_grad():
            mu = self.actor(obs_t)
            std = torch.exp(self.actor.log_std.detach())
            a = (mu + torch.randn_like(mu) * std).clamp(-1.0, 1.0)
            logp = -0.5 * (((a - mu) / std) ** 2).sum(-1) - 0.5 * np.log(2*np.pi) * a.shape[-1] - torch.log(std).sum(-1)
        return a, logp

    def store(self, idx, obs, priv, actions, logp, rewards, dones, values):
        b = self.buffer
        b.obs[idx] = obs
        b.priv[idx] = priv
        b.actions[idx] = actions
        b.log_probs[idx] = logp
        b.rewards[idx] = rewards
        b.dones[idx] = dones
        b.values[idx] = values

    def update(self, last_value):
        b = self.buffer
        b.compute_returns(last_value, self.cfg.gamma, self.cfg.lam)
        obs = b.obs.reshape(-1, b.obs.shape[-1])
        priv = b.priv.reshape(-1, b.priv.shape[-1])
        actions = b.actions.reshape(-1, b.actions.shape[-1])
        returns = b.returns.reshape(-1)
        adv = b.advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        mb = n // self.cfg.num_minibatches
        for _ in range(self.cfg.num_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.cfg.num_minibatches):
                idx = perm[i*mb:(i+1)*mb]
                mu, std = self.actor.get_dist(obs[idx])
                dist_a = actions[idx]
                logp = -0.5 * (((dist_a - mu)/std)**2).sum(-1) - 0.5*np.log(2*np.pi)*dist_a.shape[-1] - torch.log(std).sum(-1)
                old_logp = b.log_probs.reshape(-1)[idx].detach()
                ratio = torch.exp(logp - old_logp)
                clip_adv = torch.clamp(ratio, 1-self.cfg.clip, 1+self.cfg.clip) * adv[idx]
                loss_actor = -(torch.min(ratio * adv[idx], clip_adv)).mean()
                entropy = (0.5 + 0.5*np.log(2*np.pi) + torch.log(std)).sum(-1).mean()
                value = self.critic(priv[idx])
                loss_critic = 0.5 * ((value - returns[idx])**2).mean()
                loss = loss_actor - self.cfg.entropy_coef * entropy + self.cfg.value_coef * loss_critic
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.cfg.max_grad_norm)
                self.optim.step()
        self.it += 1
        # STD CAP (2026-08-15 01:05): hard stages repeatedly inflated std 0.45->0.67,
        # hedging on noisy returns -> destabilized lower stages (cascading regresses).
        # rsl_rl converges std to ~0.1-0.3; cap at 0.5 keeps exploring but exploits learned skills.
        self.actor.log_std.data.clamp_(max=float(np.log(0.5)))
        return {"loss_actor": loss_actor.item(), "loss_critic": loss_critic.item(),
                "entropy": entropy.item(), "mean_std": float(torch.exp(self.actor.log_std).mean().item())}

    def save(self, path, extra=None):
        d = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
             "optim": self.optim.state_dict(), "it": self.it}
        if extra:
            d.update(extra)
        torch.save(d, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.optim.load_state_dict(ck["optim"])
        self.it = ck.get("it", 0)
        return ck
