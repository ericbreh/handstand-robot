import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import datetime
import time 

# Import the environment class
from envs.five_link_env import FiveLinkCartwheelEnv

# --- 1. REGISTER THE ENVIRONMENT ---
register(
    id="FiveLinkCartwheel-v0",
    entry_point="envs.five_link_env:FiveLinkCartwheelEnv",
    max_episode_steps=1000, 
)

# --- CUSTOM CALLBACK TO SAVE NORM STATS PERIODICALLY ---
class SaveEnvStatsCallback(BaseCallback):
    """
    Saves the VecNormalize statistics every `save_freq` steps.
    This allows us to play intermediate checkpoints even if training crashes.
    """
    def __init__(self, save_path, save_freq=100_000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
            # self.training_env accesses the vectorized env attached to the model
            self.training_env.save(stats_path)
            if self.verbose > 0:
                print(f"Saved VecNormalize stats to {stats_path}")
        return True

def train_agent():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_Cartwheel_{run_id}"
    base_dir = f"./runs/{run_name}"
    
    log_dir = os.path.join(base_dir, "logs")
    models_dir = os.path.join(base_dir, "checkpoints")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"--- Starting Run: {run_name} ---")
    print(f"All outputs will be saved to: {base_dir}")

    env_id = "FiveLinkCartwheel-v0"
    vec_env = make_vec_env(env_id, n_envs=4, seed=0, vec_env_cls=DummyVecEnv)

    # --- NORMALIZE OBSERVATIONS AND REWARDS ---
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

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

    # --- CALLBACKS ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path=models_dir, 
        name_prefix="ckpt"
    )
    
    # New callback to save stats periodically
    stats_callback = SaveEnvStatsCallback(
        save_path=base_dir, 
        save_freq=100_000
    )

    # Path for final save
    final_model_path = os.path.join(base_dir, "final_model")
    stats_path = os.path.join(base_dir, "vec_normalize.pkl")

    # --- ROBUST TRAINING LOOP ---
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            tb_log_name="PPO", 
            callback=[checkpoint_callback, stats_callback], 
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C). Saving progress...")
    finally:
        # This block runs whether the script finishes normally OR is interrupted
        model.save(final_model_path)
        vec_env.save(stats_path)
        print(f"Model saved to {final_model_path}.zip")
        print(f"Normalization stats saved to {stats_path}")
    
    # --- VISUALIZATION / TEST ---
    print("Testing trained agent...")
    vec_env.close()
    
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])
    eval_env = VecNormalize.load(stats_path, eval_env)
    
    eval_env.training = False 
    eval_env.norm_reward = False

    obs = eval_env.reset()
    print("Running visualization... Press Ctrl+C to stop.")
    
    try:
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = eval_env.step(action)
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("Visualization stopped.")
    finally:
        eval_env.close()

if __name__ == "__main__":
    train_agent()