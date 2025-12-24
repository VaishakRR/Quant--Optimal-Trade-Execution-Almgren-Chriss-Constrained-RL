
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128,128)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden, 1])
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x).squeeze(-1)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128,128), act_limit=1.0):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], activation=nn.ReLU, output_activation=nn.Tanh)
        self.act_limit = act_limit
    def forward(self, obs):
        a = self.net(obs)
        return self.act_limit * a

class CQLAgent:
    def __init__(self, obs_dim, act_dim, act_limit=1.0, device='cpu',
                 critic_lr=3e-4, actor_lr=3e-4, gamma=0.99, tau=0.005,
                 cql_alpha=1.0, cql_n_actions=8, bc_coef=0.5):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.actor = Actor(obs_dim, act_dim, act_limit=act_limit).to(self.device)
        self.critic1 = Critic(obs_dim, act_dim).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim).to(self.device)
        self.critic1_targ = Critic(obs_dim, act_dim).to(self.device)
        self.critic2_targ = Critic(obs_dim, act_dim).to(self.device)
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.bc_coef = bc_coef
        self.mse_loss = nn.MSELoss()

    def _sample_random_actions(self, obs, n):
        bs = obs.shape[0]
        rand = torch.rand(bs, n, self.act_dim, device=self.device) * 2 - 1
        return rand * self.act_limit

    def update(self, obs_np, act_np, rew_np, next_obs_np, done_np, lmbda_penalty=0.0):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act_np, dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(rew_np, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_a = self.actor(next_obs)
            next_a = torch.clamp(next_a + 0.1 * torch.randn_like(next_a), -self.act_limit, self.act_limit)
            q1_targ = self.critic1_targ(next_obs, next_a)
            q2_targ = self.critic2_targ(next_obs, next_a)
            q_targ = torch.min(q1_targ, q2_targ)
            backup = rew + lmbda_penalty + self.gamma * (1 - done) * q_targ

        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        bellman_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        batch_size = obs.shape[0]
        n = self.cql_n_actions
        random_actions = self._sample_random_actions(obs, n)
        obs_rep = obs.unsqueeze(1).repeat(1, n, 1).reshape(batch_size*n, -1)
        random_flat = random_actions.reshape(batch_size*n, -1)
        q1_rand = self.critic1(obs_rep, random_flat).reshape(batch_size, n)
        q2_rand = self.critic2(obs_rep, random_flat).reshape(batch_size, n)

        pi_actions = self.actor(obs).unsqueeze(1).repeat(1, n, 1)
        pi_actions = torch.clamp(pi_actions + 0.1 * torch.randn_like(pi_actions), -self.act_limit, self.act_limit)
        pi_flat = pi_actions.reshape(batch_size*n, -1)
        q1_pi = self.critic1(obs_rep, pi_flat).reshape(batch_size, n)
        q2_pi = self.critic2(obs_rep, pi_flat).reshape(batch_size, n)

        cat_q1 = torch.cat([q1_rand, q1_pi], dim=1)
        cat_q2 = torch.cat([q2_rand, q2_pi], dim=1)
        lse_q1 = torch.logsumexp(cat_q1, dim=1) - np.log(cat_q1.shape[1])
        lse_q2 = torch.logsumexp(cat_q2, dim=1) - np.log(cat_q2.shape[1])
        cql_penalty = (lse_q1 - q1).mean() + (lse_q2 - q2).mean()
        cql_loss = self.cql_alpha * cql_penalty

        critic_loss = bellman_loss + cql_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a_pi = self.actor(obs)
        q1_pi_for_grad = self.critic1(obs, a_pi)
        actor_loss_q = -q1_pi_for_grad.mean()
        bc_loss = self.mse_loss(a_pi, act)
        actor_loss = actor_loss_q + self.bc_coef * bc_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for p, p_targ in zip(self.critic1.parameters(), self.critic1_targ.parameters()):
            p_targ.data.mul_(1 - self.tau)
            p_targ.data.add_(self.tau * p.data)
        for p, p_targ in zip(self.critic2.parameters(), self.critic2_targ.parameters()):
            p_targ.data.mul_(1 - self.tau)
            p_targ.data.add_(self.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "bc_loss": bc_loss.item()
        }
