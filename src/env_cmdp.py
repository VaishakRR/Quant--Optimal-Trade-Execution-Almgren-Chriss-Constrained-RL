import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ExecEnv(gym.Env):
    """
    Gymnasium-style execution environment with terminal inventory penalty and
    forced liquidation on final step.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 X=1000.0,
                 T=50,
                 V=1000.0,
                 p_max=0.2,
                 dt=1.0,
                 sigma=0.02,
                 eta=1e-4,
                 gamma=1e-5,
                 terminal_penalty=0.05,
                 seed=None):
        super().__init__()
        self.X = float(X)
        self.T = int(T)
        self.V = float(V)
        self.p_max = float(p_max)
        self.dt = float(dt)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.terminal_penalty = float(terminal_penalty)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(low=-self.p_max, high=self.p_max, shape=(1,), dtype=np.float32)
        obs_low = np.array([0.0, -np.finfo(np.float32).max, 0.0, -np.finfo(np.float32).max, 0.0], dtype=np.float32)
        obs_high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.reset()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.x = float(self.X)
        self.t = 0
        self.midprice = 100.0
        self.vol_est = float(self.sigma)
        self.ofi = 0.0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.x / max(1.0, self.X),
            self.midprice,
            self.vol_est,
            self.ofi,
            self.t / max(1.0, self.T)
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called on finished env; call reset().")

        p = float(np.clip(np.asarray(action).item(), -self.p_max, self.p_max))
        V_t = self.V * (1.0 + 0.1 * self.rng.standard_normal())

        
        last_step = (self.t >= self.T - 1)
        if last_step and self.x > 0:
            exec_vol = self.x  
        else:
            exec_vol = max(0.0, p * V_t)
            exec_vol = min(exec_vol, self.x)

        exec_price = self.midprice - self.eta * (exec_vol / max(1e-6, self.dt))
        proceeds = exec_vol * exec_price

    
        self.midprice = (self.midprice
                         - self.gamma * (exec_vol / max(1e-6, self.dt)) * self.dt
                         + self.vol_est * np.sqrt(self.dt) * self.rng.standard_normal())

        self.x -= exec_vol
        self.t += 1

        instant_cost = (self.midprice + self.gamma * (exec_vol / max(1e-6, self.dt)) * self.dt) * exec_vol - proceeds
        reward = -float(instant_cost)

        self.ofi = 0.95 * self.ofi + 0.05 * (exec_vol - (V_t - exec_vol))
        self.vol_est = 0.98 * self.vol_est + 0.02 * abs(self.midprice - exec_price) / max(1e-6, np.sqrt(self.dt))

        terminated = bool(self.t >= self.T or self.x <= 0.0)
        truncated = False

        if terminated:
            leftover = max(0.0, self.x)
            reward = float(reward) - self.terminal_penalty * leftover

        info = {
            "exec_vol": exec_vol,
            "exec_price": exec_price,
            "midprice": self.midprice,
            "instant_cost": instant_cost,
            "remaining_inventory": self.x
        }

        if terminated:
            self.done = True

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode='human'):
        print(f"t={self.t}, x={self.x:.2f}, mid={self.midprice:.4f}")
