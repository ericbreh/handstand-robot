import os
import sys
import gymnasium as gym

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.handstand_env import HandstandEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    # 1. Create Vectorized Environment (Parallel Training)
    # This runs 4 simulations at once. 
    # It speeds up training and makes the learning much more stable.
    n_envs = 4
    
    # We pass the class directly. make_vec_env handles the rest.
    env = make_vec_env(
        HandstandEnv, 
        n_envs=n_envs, 
        seed=42,
        vec_env_cls=SubprocVecEnv # Runs in separate processes
    )

    # 2. Define the PPO Model (Tuned for Robotics)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        
        # Hyperparameters Tuned for Continuous Control:
        learning_rate=3e-4,     # Standard, effective
        n_steps=2048,           # Steps per update (per env)
        batch_size=256,         # Larger batch = smoother gradient
        ent_coef=0.005,         # EXPLORATION! Forces robot to try new things.
        
        tensorboard_log="./runs/"
    )

    # 3. Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, # Note: This counts total steps across all envs
        save_path='./runs/checkpoints/',
        name_prefix='handstand_model'
    )

    # 4. Train
    print(f"Starting training on {n_envs} parallel environments...")
    print("Goal: Imitate the expert trajectory.")
    
    # 2 Million steps is a good target. 
    # With 4 envs, this will finish 4x faster than your previous script.
    model.learn(total_timesteps=15_000_000, callback=checkpoint_callback)

    # 5. Save Final
    model.save("handstand_final")
    print("Training finished. Model saved.")
    
    # Close parallel processes
    env.close()