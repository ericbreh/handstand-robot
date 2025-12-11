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
    # Create vectorized environment for parallel training
    n_envs = 4
    
    env = make_vec_env(
        HandstandEnv, 
        n_envs=n_envs, 
        seed=42,
        vec_env_cls=SubprocVecEnv
    )

    # Define PPO model with hyperparameters tuned for robotics
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        ent_coef=0.005,
        tensorboard_log="./runs/"
    )

    # Setup checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path='./runs/checkpoints/',
        name_prefix='handstand_model'
    )

    # Start training
    print(f"Starting training on {n_envs} parallel environments...")
    print("Goal: Imitate the expert trajectory.")
    
    model.learn(total_timesteps=15_000_000, callback=checkpoint_callback)

    # Save final model
    model.save("handstand_final")
    print("Training finished. Model saved.")
    
    env.close()