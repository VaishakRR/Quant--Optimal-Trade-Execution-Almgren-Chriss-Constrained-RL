import numpy as np
from env_cmdp import ExecEnv

class CMDPEnv:
    """
    Wrapper around ExecEnv that tracks episode-level constraints and returns
    shaped rewards for Lagrangian primal-dual optimization.
    """
    def __init__(self, X=500.0, T=30, V=800.0, terminal_penalty=0.1,
                 p_max=0.2, constraint_config=None):
        self.env = ExecEnv(X=X, T=T, V=V, seed=None, terminal_penalty=terminal_penalty)
        self.p_max = p_max
        self.constraint_config = constraint_config or {}
        self.reset_metrics()

    def reset_metrics(self):
        self.ep_actions = []
        self.ep_exec = []
        self.ep_costs = []
        self.ep_x = []
        self.tot_cost = 0.0

    def reset(self, seed=None):
        self.reset_metrics()
        obs = self.env.reset(seed=seed)
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs

    def step(self, action, lambda_vec=None):
        p = float(np.clip(np.asarray(action).item(), -self.p_max, self.p_max))
        res = self.env.step(np.array([p], dtype=np.float32))
        if len(res) == 4:
            obs, reward, done, info = res
        else:
            obs, reward, term, trunc, info = res
            done = bool(term or trunc)
        self.ep_actions.append(p)
        self.ep_exec.append(float(info.get('exec_vol', 0.0)))
        self.ep_costs.append(float(info.get('instant_cost', 0.0)))
        self.ep_x.append(float(info.get('remaining_inventory', getattr(self.env, 'x', 0.0))))
        self.tot_cost += float(info.get('instant_cost', 0.0))

        lagr_pen = 0.0
        if lambda_vec:
            part_limit = self.constraint_config.get('participation_mean', {}).get('limit', None)
            if part_limit is not None:
                g_part = abs(p) / (part_limit + 1e-12)  
                lagr_pen += lambda_vec.get('participation_mean', 0.0) * (g_part - 1.0/ (self.env.T + 0.0))

        shaped_reward = reward - float(lagr_pen)
        return obs, shaped_reward, done, info

    def finalize_episode_constraints(self):
        """
        Compute episode-level constraint values g_i for primal-dual update.
        Returns dict of constraint_name -> value (raw, not normalized).
        """
        vals = {}
        if 'participation_mean' in self.constraint_config:
            vals['participation_mean'] = float(np.mean(np.abs(self.ep_actions)))
        if 'terminal_shortfall' in self.constraint_config:
            vals['terminal_shortfall'] = float(self.ep_x[-1] if len(self.ep_x)>0 else 0.0)
        if 'mean_cost' in self.constraint_config:
            vals['mean_cost'] = float(np.mean(self.ep_costs) if len(self.ep_costs)>0 else 0.0)
        return vals

    def render(self): 
        self.env.render()
