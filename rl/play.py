import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.envs.registration import register
import os
import glob
import time

# 1. Register the environment (must match training)
register(
    id="FiveLinkCartwheel-v0",
    entry_point="envs.five_link_env:FiveLinkCartwheelEnv",
    max_episode_steps=1000,
)


def get_latest_run_dir(runs_dir="./runs"):
    """Finds the directory with the latest timestamp."""
    if not os.path.exists(runs_dir):
        print(f"Error: Directory '{runs_dir}' not found. Have you trained yet?")
        return None

    # Get all subdirectories in runs/
    all_runs = [
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]

    if not all_runs:
        print("Error: No run directories found.")
        return None

    # Sort by creation time (latest last)
    latest_run = max(all_runs, key=os.path.getmtime)
    return latest_run


def play_latest():
    latest_dir = get_latest_run_dir()
    if not latest_dir:
        return

    print(f"Loading from latest run: {latest_dir}")

    # Paths to files
    final_model_path = os.path.join(latest_dir, "final_model.zip")
    stats_path = os.path.join(latest_dir, "vec_normalize.pkl")

    # Check if final model exists, otherwise look for checkpoints
    model_to_load = final_model_path
    if not os.path.exists(final_model_path):
        print("Final model not found. Checking for checkpoints...")
        ckpt_dir = os.path.join(latest_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            ckpts = glob.glob(os.path.join(ckpt_dir, "*.zip"))
            if ckpts:
                # Load the latest checkpoint based on step count in filename
                # Filename format is usually name_prefix_steps_steps.zip
                # Simple sort by modification time is usually sufficient
                model_to_load = max(ckpts, key=os.path.getmtime)
                print(f"Found checkpoint: {model_to_load}")
            else:
                print("No checkpoints found.")
                return
        else:
            print("No checkpoints directory found.")
            return

    if not os.path.exists(stats_path):
        print(
            "Error: vec_normalize.pkl not found. Cannot run model correctly without normalization stats."
        )
        return

    # --- SETUP ENVIRONMENT ---
    env_id = "FiveLinkCartwheel-v0"

    # We must wrap the env in DummyVecEnv because VecNormalize expects it
    env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])

    # Load the normalization statistics
    print(f"Loading normalization stats from {stats_path}...")
    env = VecNormalize.load(stats_path, env)

    # IMPORTANT: Don't update stats or normalize rewards during test
    env.training = False
    env.norm_reward = False

    # Load the agent
    print(f"Loading model from {model_to_load}...")
    model = PPO.load(model_to_load)

    # --- RUN LOOP ---
    print("\n--- Playing Model (Press Ctrl+C to stop) ---")
    obs = env.reset()

    try:
        while True:
            # Deterministic=True generally performs better for testing
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            env.render()
            # Slow down slightly to make it watchable (MuJoCo is very fast)
            # time.sleep(1.0 / 40.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    play_latest()
