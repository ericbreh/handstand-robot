# import gymnasium as gym
# from gymnasium.envs.registration import register
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
# import os
# import datetime
# import time
# import glob

# # Import the environment class
# from envs.five_link_env import FiveLinkCartwheelEnv

# # --- CONFIGURATION ---
# CONTINUE_FROM_LATEST = True 
# TOTAL_TIMESTEPS = 5_000_000 

# register(
#     id="FiveLinkCartwheel-v0",
#     entry_point="envs.five_link_env:FiveLinkCartwheelEnv",
#     max_episode_steps=1000, 
# )

# class SaveEnvStatsCallback(BaseCallback):
#     def __init__(self, save_path, save_freq=100_000, verbose=1):
#         super().__init__(verbose)
#         self.save_path = save_path
#         self.save_freq = save_freq

#     def _on_step(self) -> bool:
#         if self.n_calls % self.save_freq == 0:
#             stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
#             self.training_env.save(stats_path)
#         return True

# def get_previous_run(runs_dir="./runs", current_run_name=None):
#     """
#     Finds the latest run EXCLUDING the current one we just created.
#     """
#     if not os.path.exists(runs_dir): return None
    
#     # Get all subdirectories
#     all_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    
#     # Filter out the current run directory if it exists in the list
#     valid_runs = [r for r in all_runs if current_run_name not in r]
    
#     if not valid_runs: return None
    
#     # Return the newest of the remaining runs
#     return max(valid_runs, key=os.path.getmtime)

# def train_agent():
#     # 1. Setup New Run Directory FIRST
#     run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_name = f"PPO_Cartwheel_{run_id}"
#     base_dir = f"./runs/{run_name}"
    
#     log_dir = os.path.join(base_dir, "logs")
#     models_dir = os.path.join(base_dir, "checkpoints")
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(models_dir, exist_ok=True)

#     print(f"--- STARTING NEW RUN: {run_name} ---")

#     # 2. Create Environment
#     env_id = "FiveLinkCartwheel-v0"
#     vec_env = make_vec_env(env_id, n_envs=4, seed=0, vec_env_cls=DummyVecEnv)

#     # 3. Handle Resuming
#     model = None
    
#     # Pass the current run name so we don't accidentally try to resume from ourselves
#     previous_run = get_previous_run(current_run_name=run_name)
    
#     if CONTINUE_FROM_LATEST and previous_run:
#         print(f"--- FOUND PREVIOUS RUN: {previous_run} ---")
        
#         # A. Load Normalization Stats
#         stats_path = os.path.join(previous_run, "vec_normalize.pkl")
#         if os.path.exists(stats_path):
#             print(f"Loading VecNormalize stats from previous run...")
#             vec_env = VecNormalize.load(stats_path, vec_env)
#             vec_env.training = True 
#             vec_env.norm_reward = True
#         else:
#             print("Warning: No stats found in previous run. Starting normalization from scratch.")
#             vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

#         # B. Load Model (Final or Checkpoint)
#         model_path = os.path.join(previous_run, "final_model.zip")
#         if not os.path.exists(model_path):
#             print("Final model not found, checking checkpoints...")
#             ckpt_dir = os.path.join(previous_run, "checkpoints")
#             if os.path.exists(ckpt_dir):
#                 ckpts = glob.glob(os.path.join(ckpt_dir, "*.zip"))
#                 if ckpts: 
#                     model_path = max(ckpts, key=os.path.getmtime)
#                     print(f"Found checkpoint: {model_path}")
        
#         if os.path.exists(model_path):
#             print(f"Loading weights from: {model_path}")
#             model = PPO.load(model_path, env=vec_env, tensorboard_log=log_dir)
#             print("Model loaded successfully.")
#         else:
#             print("No model file found to resume. Starting fresh.")
#             model = None
#     else:
#         print("No previous run found or resume disabled. Starting fresh.")
#         vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

#     # 4. Initialize New Model (if not loaded)
#     if model is None:
#         model = PPO(
#             "MlpPolicy", 
#             vec_env, 
#             verbose=1, 
#             learning_rate=3e-4, 
#             n_steps=2048,
#             batch_size=64,
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#             clip_range=0.2,
#             ent_coef=0.01, 
#             tensorboard_log=log_dir 
#         )

#     print(f"Training for {TOTAL_TIMESTEPS/1e6:.1f}M steps...")
    
#     callbacks = [
#         CheckpointCallback(save_freq=100_000, save_path=models_dir, name_prefix="ckpt"),
#         SaveEnvStatsCallback(save_path=base_dir, save_freq=100_000)
#     ]

#     try:
#         model.learn(
#             total_timesteps=TOTAL_TIMESTEPS, 
#             tb_log_name="PPO", 
#             callback=callbacks, 
#             reset_num_timesteps=False 
#         )
#     except KeyboardInterrupt:
#         print("\nTraining interrupted.")
#     finally:
#         final_model_path = os.path.join(base_dir, "final_model")
#         stats_path = os.path.join(base_dir, "vec_normalize.pkl")
#         model.save(final_model_path)
#         vec_env.save(stats_path)
#         print(f"Saved final model and stats to {base_dir}")
#         vec_env.close()

# if __name__ == "__main__":
#     train_agent()

    
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
    max_episode_steps=400, 
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

    total_timesteps = 15_000_000
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