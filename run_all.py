import sys
import os
import torch
import numpy as np

# --- THE FIX: Add 'src' to the Python path ---
# This tells Python: "Look in the 'src' folder for files too"
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
# ---------------------------------------------

# Now these imports will work
from env_cmdp import ExecEnv
from cmdp_env import CMDPEnv
from iql import IQLAgent
from trainer import CMDP_IQL_Trainer
from dataset import OfflineReplayBuffer

def main():
    print("--- Starting Full Pipeline Run ---")
    
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 2. Check Paths
    root_dir = current_dir
    chk_dir = os.path.join(root_dir, 'checkpoints')
    os.makedirs(chk_dir, exist_ok=True)
    
    # 3. Environment
    print("Initializing Environment...")
    env = CMDPEnv(X=500.0, T=30, V=800.0, terminal_penalty=0.1, 
                  constraint_config={'participation_mean': {'limit': 0.05}})
    
    # 4. Generate/Load Data
    buffer_path = os.path.join(chk_dir, 'replay_buffer.pkl')
    
    if os.path.exists(buffer_path):
        print(f"Loading existing buffer from {buffer_path}")
        import pickle
        with open(buffer_path, 'rb') as f:
            buffer = pickle.load(f)
    else:
        print("Generating fresh data (TWAP)...")
        # Quick generation if missing
        buffer = OfflineReplayBuffer(env.env.observation_space.shape[0], 1, max_size=10000, device=device)
        # (Simplified generation logic would go here if needed, but usually we assume it exists)
        # For this script, we assume the buffer exists or we fail gracefully
        raise FileNotFoundError("Replay buffer not found! Please run the generator step first.")

    # 5. Agent
    print("Initializing IQL Agent...")
    agent = IQLAgent(state_dim=env.env.observation_space.shape[0], 
                     action_dim=1, 
                     act_limit=env.p_max, 
                     device=device)
    
    # 6. Trainer
    trainer = CMDP_IQL_Trainer(env=env, agent=agent, buffer=buffer, device=device)
    
    # 7. Run
    print("Starting BC Warmup...")
    trainer.train_bc_only(steps=500, log_every=100)
    
    print("Starting IQL Training...")
    trainer.train_offline_loop(steps=1000, eval_every=200, update_duals_every=200)
    
    # 8. Save
    save_path = os.path.join(chk_dir, 'agent_final.pth')
    agent.save(save_path)
    print(f"Run complete. Agent saved to {save_path}")

if __name__ == "__main__":
    main()
