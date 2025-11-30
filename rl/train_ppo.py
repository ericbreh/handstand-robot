import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback 
import os
import datetime
import time # Added for visualization timing

# Import the environment class
from envs.five_link_env import FiveLinkCartwheelEnv

# --- 1. REGISTER THE ENVIRONMENT ---
register(
    id="FiveLinkCartwheel-v0",
    entry_point="envs.five_link_env:FiveLinkCartwheelEnv",
    max_episode_steps=1000, 
)

def train_agent():
    """
    Sets up the vectorized environment and trains a PPO agent.
    Saves everything into a unique timestamped folder for easy comparison.
    """
    # Create a unique ID for this run based on the current time
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_Cartwheel_{run_id}"

    # Define the main directory for this specific run
    base_dir = f"./runs/{run_name}"
    
    # Define sub-directories for logs and checkpoints within this run folder
    log_dir = os.path.join(base_dir, "logs")
    models_dir = os.path.join(base_dir, "checkpoints")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"--- Starting Run: {run_name} ---")
    print(f"All outputs will be saved to: {base_dir}")

    # We use make_vec_env to create a vectorized environment
    env_id = "FiveLinkCartwheel-v0"
    vec_env = make_vec_env(env_id, n_envs=4, seed=0, vec_env_cls=DummyVecEnv)

    # --- CRITICAL: NORMALIZE OBSERVATIONS AND REWARDS ---
    # This is standard practice for MuJoCo tasks. It scales inputs to mean 0, std 1.
    # It significantly speeds up convergence.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Initialize the PPO agent
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, 
        tensorboard_log=log_dir 
    )

    total_timesteps = 5_000_000
    print(f"Starting training for {total_timesteps/1e6:.1f}M timesteps...")
    print(f"To view logs, run: tensorboard --logdir ./runs")

    # --- CHECKPOINT CALLBACK ---
    # We also need to save the VecNormalize statistics alongside the model
    # We create a custom saving loop or just save the final one. 
    # For simplicity, we stick to the basic checkpoint for the model weights.
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path=models_dir, 
        name_prefix="ckpt"
    )

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps, 
        tb_log_name="PPO", 
        callback=checkpoint_callback, 
        reset_num_timesteps=False
    )

    # Save the final trained model
    final_model_path = os.path.join(base_dir, "final_model")
    model.save(final_model_path)
    
    # Save the Normalization Statistics
    # We MUST save this, otherwise the agent won't know how to scale inputs during testing
    stats_path = os.path.join(base_dir, "vec_normalize.pkl")
    vec_env.save(stats_path)
    
    print(f"Training complete.")
    print(f"Model saved to {final_model_path}.zip")
    print(f"Normalization stats saved to {stats_path}")
    
    # --- VISUALIZATION / TEST ---
    print("Testing trained agent...")
    vec_env.close()
    
    # To visualize, we must replicate the training environment EXACTLY
    # 1. Create a dummy vec env (Gym needs this for VecNormalize)
    # 2. Load the saved normalization stats
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])
    eval_env = VecNormalize.load(stats_path, eval_env)
    
    # IMPORTANT: Turn off training updates and reward normalization during test
    eval_env.training = False 
    eval_env.norm_reward = False

    obs = eval_env.reset()
    print("Running visualization... Press Ctrl+C to stop.")
    
    try:
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = eval_env.step(action)
            # Slow down slightly to match real-time (approx 50fps)
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("Visualization stopped.")
    finally:
        eval_env.close()

if __name__ == "__main__":
    train_agent()