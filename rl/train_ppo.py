import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import datetime
import time 
import glob

# Import the environment class
from envs.five_link_env import FiveLinkCartwheelEnv

# --- 1. REGISTER THE ENVIRONMENT ---
register(
    id="FiveLinkCartwheel-v0",
    entry_point="envs.five_link_env:FiveLinkCartwheelEnv",
    max_episode_steps=100, 
)

# --- CONFIGURATION ---
CONTINUE_FROM_LATEST = False  # <--- CHANGED: Forces a fresh start
TOTAL_TIMESTEPS = 1_000_000 
SAVE_FREQ = 100_000         

# --- CUSTOM CALLBACK TO SAVE MATCHED STATS ---
class SaveMatchedStatsCallback(BaseCallback):
    """
    Saves the VecNormalize stats with the EXACT same naming convention 
    and frequency as the CheckpointCallback.
    """
    def __init__(self, save_path, save_freq, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 1. Save "Latest" (for quick resuming/playing)
            latest_path = os.path.join(self.save_path, "vec_normalize.pkl")
            self.training_env.save(latest_path)
            
            # 2. Save "Versioned" to match the checkpoint exactly
            ckpt_dir = os.path.join(self.save_path, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            versioned_filename = f"vec_normalize_{self.n_calls*4.0}_steps.pkl"
            versioned_path = os.path.join(ckpt_dir, versioned_filename)
            
            self.training_env.save(versioned_path)
            
            if self.verbose > 0:
                print(f"Synced Save: {versioned_filename}")
        return True

def get_previous_run(runs_dir="./runs", current_run_name=None):
    if not os.path.exists(runs_dir): return None
    all_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    valid_runs = [r for r in all_runs if current_run_name not in r]
    if not valid_runs: return None
    return max(valid_runs, key=os.path.getmtime)

def train_agent():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_Cartwheel_{run_id}"
    base_dir = f"./runs/{run_name}"
    
    log_dir = os.path.join(base_dir, "logs")
    models_dir = os.path.join(base_dir, "checkpoints")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"--- STARTING NEW RUN: {run_name} ---")

    env_id = "FiveLinkCartwheel-v0"
    vec_env = make_vec_env(env_id, n_envs=10, seed=0, vec_env_cls=SubprocVecEnv)

    model = None
    previous_run = get_previous_run(current_run_name=run_name)
    
    if CONTINUE_FROM_LATEST and previous_run:
        print(f"--- FOUND PREVIOUS RUN: {previous_run} ---")
        # (Resume logic skipped since CONTINUE is False)
    else:
        # Start Fresh: Create new normalization wrapper
        print("Starting training from scratch (No previous run loaded).")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    if model is None:
        model = PPO(
            "MlpPolicy",
            vec_env, 
            verbose=1, 
            learning_rate=3e-4, 
            n_steps=2048 // 8,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01, 
            tensorboard_log=log_dir,
            device="cpu"
        )

    print(f"Training for {TOTAL_TIMESTEPS/1e6:.1f}M steps...")
    print(f"Saving synced checkpoints every {SAVE_FREQ} steps.")
    
    callbacks = [
        CheckpointCallback(save_freq=SAVE_FREQ, save_path=models_dir, name_prefix="ckpt"),
        SaveMatchedStatsCallback(save_path=base_dir, save_freq=SAVE_FREQ) 
    ]

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            tb_log_name="PPO", 
            callback=callbacks, 
            reset_num_timesteps=False 
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        final_model_path = os.path.join(base_dir, "final_model")
        stats_path = os.path.join(base_dir, "vec_normalize.pkl")
        model.save(final_model_path)
        vec_env.save(stats_path)
        print(f"Saved final model and stats to {base_dir}")
        vec_env.close()

if __name__ == "__main__":
    train_agent()