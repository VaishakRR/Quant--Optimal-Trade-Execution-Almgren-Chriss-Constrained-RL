import torch
import numpy as np
import os
import time
from collections import defaultdict
import copy

class CMDP_IQL_Trainer:
    def __init__(
        self,
        env,              
        agent,            
        buffer,           
        batch_size=256,
        lambda_lr=0.05,   
        lambda_max=100.0, 
        device="cpu"
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.batch_size = batch_size
        self.device = device
        
        self.lambda_vec = {k: 0.0 for k in self.env.constraint_config.keys()}
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max

        self.logs = defaultdict(list)

    def compute_batch_penalties(self, actions):
        """
        Approximate constraint costs for a batch of actions to shape rewards.
        Ensure output shape is [batch_size, 1] to match rewards.
        """
       
        penalties = torch.zeros_like(actions)
        
        if 'participation_mean' in self.env.constraint_config:
            limit = self.env.constraint_config['participation_mean']['limit']
            lam = self.lambda_vec.get('participation_mean', 0.0)
            if lam > 0:
                g_proxy = torch.abs(actions) / (limit + 1e-6)
                penalties += lam * g_proxy / self.env.env.T 
        
        return penalties

    def train_bc_only(self, steps=5000, log_every=1000):
        """
        Phase 1: Warm-start the actor using Behavior Cloning.
        """
        print(f"--- Starting BC Warm-up ({steps} steps) ---")
        self.agent.actor.train()
        
        for i in range(steps):
            s, a, _, _, _ = self.buffer.sample(self.batch_size)
            s = s.to(self.device)
            a = a.to(self.device)
            
            pred_action, _ = self.agent.actor(s)
            bc_loss = ((pred_action - a)**2).mean()
            
            self.agent.actor_optimizer.zero_grad()
            bc_loss.backward()
            self.agent.actor_optimizer.step()
            
            if (i+1) % log_every == 0:
                print(f"BC Step {i+1}: Loss = {bc_loss.item():.5f}")
                self.logs['bc_loss'].append(bc_loss.item())

        print("--- BC Warm-up Complete ---")

    def evaluate_policy(self, n_episodes=5):
        """
        Run episodes to measure constraint violations for Dual Update.
        """
        self.agent.actor.eval()
        vals = []
        exec_vols = []
        
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    a = self.agent.actor.get_action(
                        torch.as_tensor(obs.reshape(1,-1), dtype=torch.float32).to(self.device),
                        deterministic=True
                    )
                    a = a.cpu().numpy().reshape(-1)[0]
                
                obs, _, done, info = self.env.step(a, lambda_vec=self.lambda_vec)
            
            vals.append(self.env.finalize_episode_constraints())
            exec_vols.append(self.env.ep_exec)

        mean_constraints = {k: np.mean([v.get(k, 0.0) for v in vals]) for k in vals[0].keys()}
        mean_exec = np.sum([np.sum(e) for e in exec_vols]) / n_episodes
        
        self.agent.actor.train()
        return mean_constraints, mean_exec

    def update_duals(self, mean_constraints):
        """
        Projected Gradient Ascent on Lagrangians.
        """
        for k, current_val in mean_constraints.items():
            limit = self.env.constraint_config.get(k, {}).get('limit', 0.0)
            grad = current_val - limit
            
            new_lambda = self.lambda_vec.get(k, 0.0) + self.lambda_lr * grad
            new_lambda = max(0.0, min(self.lambda_max, new_lambda))
            
            self.lambda_vec[k] = new_lambda
        
        return self.lambda_vec

    def train_offline_loop(self, steps=10000, eval_every=500, update_duals_every=500):
        """
        Phase 2: Offline RL (IQL) with Primal-Dual updates.
        """
        print(f"--- Starting IQL Fine-tuning ({steps} steps) ---")
        
        for i in range(steps):
            s, a, ns, r, d = self.buffer.sample(self.batch_size)
            s, a, ns, r, d = s.to(self.device), a.to(self.device), ns.to(self.device), r.to(self.device), d.to(self.device)

            with torch.no_grad():
                penalties = self.compute_batch_penalties(a)
                if penalties.ndim == 1: penalties = penalties.unsqueeze(-1)
                
                r_shaped = r - penalties
            
            with torch.no_grad():
                q1, q2 = self.agent.q_target(s, a)
                q_min = torch.min(q1, q2)
            
            v_pred = self.agent.value_net(s)
            v_loss = self.agent.expectile_loss(q_min - v_pred).mean()
            self.agent.v_optimizer.zero_grad()
            v_loss.backward()
            self.agent.v_optimizer.step()

            with torch.no_grad():
                next_v = self.agent.value_net(ns)
                q_target_val = r_shaped + self.agent.discount * d * next_v.detach()
            
            q1_pred, q2_pred = self.agent.q_critic(s, a)
            q_loss = torch.nn.functional.mse_loss(q1_pred, q_target_val) + torch.nn.functional.mse_loss(q2_pred, q_target_val)
            self.agent.q_optimizer.zero_grad()
            q_loss.backward()
            self.agent.q_optimizer.step()

            with torch.no_grad():
                q1, q2 = self.agent.q_target(s, a)
                q_val = torch.min(q1, q2)
                v_val = self.agent.value_net(s)
                advantage = q_val - v_val
                exp_adv = torch.exp(advantage * self.agent.temperature).clamp(max=100.0)
            
            mean, std = self.agent.actor(s)
            normal = torch.distributions.Normal(mean, std)
            log_prob = normal.log_prob(a).sum(dim=-1, keepdim=True)
            actor_loss = -(exp_adv * log_prob).mean()
            
            self.agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.agent.actor_optimizer.step()

            for param, target_param in zip(self.agent.q_critic.parameters(), self.agent.q_target.parameters()):
                target_param.data.copy_(self.agent.tau * param.data + (1 - self.agent.tau) * target_param.data)
            
            if (i+1) % update_duals_every == 0:
                constraints, exec_vol = self.evaluate_policy()
                self.update_duals(constraints)
                print(f"Step {i+1} | Constraints: {constraints} | Lambdas: {self.lambda_vec}")
                self.logs['constraints'].append(constraints)
                self.logs['lambdas'].append(copy.deepcopy(self.lambda_vec))
                self.logs['exec_vol'].append(exec_vol)

            if (i+1) % 100 == 0:
                self.logs['q_loss'].append(q_loss.item())
                self.logs['v_loss'].append(v_loss.item())

        print("--- IQL Fine-tuning Complete ---")
        return self.logs
