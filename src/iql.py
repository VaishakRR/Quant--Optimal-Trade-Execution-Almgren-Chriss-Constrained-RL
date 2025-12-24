import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dim)
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.net = MLP(state_dim, hidden_dim * 2, hidden_dim) 
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, state, deterministic=False):
        mean, std = self(state)
        if deterministic:
            return torch.tanh(mean) * self.max_action
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        return y_t * self.max_action, normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)

class IQLAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        act_limit,
        hidden_dim=256,
        discount=0.99,
        tau=0.005,
        expectile=0.7,   
        temperature=3.0, 
        lr=3e-4,
        device="cpu"
    ):
        self.act_limit = act_limit
        self.device = device
        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature

        self.q_critic = TwinQ(state_dim, action_dim, hidden_dim).to(device)
        self.q_target = copy.deepcopy(self.q_critic)
        self.value_net = MLP(state_dim, 1, hidden_dim).to(device) # V function
        self.actor = GaussianActor(state_dim, action_dim, act_limit, hidden_dim).to(device)

    
        self.q_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def expectile_loss(self, diff):
        weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
        return weight * (diff**2)

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)

       
        with torch.no_grad():
            q1, q2 = self.q_target(state, action)
            q_min = torch.min(q1, q2)
        
        v_pred = self.value_net(state)
        v_loss = self.expectile_loss(q_min - v_pred).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

       
        with torch.no_grad():
            next_v = self.value_net(next_state)
            q_target_val = reward + self.discount * not_done * next_v.detach()

        q1_pred, q2_pred = self.q_critic(state, action)
        q_loss = F.mse_loss(q1_pred, q_target_val) + F.mse_loss(q2_pred, q_target_val)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        
        with torch.no_grad():
            q1, q2 = self.q_target(state, action)
            q_val = torch.min(q1, q2)
            v_val = self.value_net(state)
            advantage = q_val - v_val
            
            exp_adv = torch.exp(advantage * self.temperature)
            exp_adv = torch.clamp(exp_adv, max=100.0) 
        mean, std = self.actor(state)
        normal = torch.distributions.Normal(mean, std)
        
        
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        actor_loss = -(exp_adv * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        for param, target_param in zip(self.q_critic.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "avg_adv": advantage.mean().item()
        }

    def save(self, filename):
        torch.save({
            'q_critic': self.q_critic.state_dict(),
            'value_net': self.value_net.state_dict(),
            'actor': self.actor.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_critic.load_state_dict(checkpoint['q_critic'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.q_target = copy.deepcopy(self.q_critic)
