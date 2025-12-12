import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.envs.registration import register
import os
import glob
import time
import argparse

# 1. Register the environment (must match training)
register(
    id="FiveLinkCartwheel-v0",
    entry_point="envs.five_link_env:FiveLinkCartwheelEnv",
    max_episode_steps=400,
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


def play_latest(checkpoint_path=None):
    latest_dir = get_latest_run_dir()
    if not latest_dir:
        return

    print(f"Loading from latest run: {latest_dir}")

    # Paths to files
    final_model_path = os.path.join(latest_dir, "final_model.zip")
    stats_path = os.path.join(latest_dir, "vec_normalize.pkl")

    # If a specific checkpoint is provided, use it
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            print(f"Error: Specified checkpoint '{checkpoint_path}' not found.")
            return
        model_to_load = checkpoint_path
        print(f"Using specified checkpoint: {model_to_load}")
    # Check if final model exists, otherwise look for checkpoints
    elif os.path.exists(final_model_path):
        model_to_load = final_model_path
    else:
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

            # Access underlying MuJoCo env through the wrappers
            base_env = env.venv.envs[0].unwrapped

            obs, reward, done, info = env.step(action)
            print(f"Reward: {reward[0]:.3f}")
            if info[0]:
                print("Reward Components:")
                for key, value in info[0].items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")

            time.sleep(1.0 / 40.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play a trained PPO agent for the FiveLinkCartwheel environment."
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="Path to a specific checkpoint file to play.",
    )
    args = parser.parse_args()

    play_latest(checkpoint_path=args.checkpoint)
