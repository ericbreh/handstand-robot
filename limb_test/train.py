import os
import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


torch.set_num_threads(8)

# 1. Create unique log directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./runs/limb_spin_{timestamp}"
print(f"Logging to: {log_dir}")


# 2. Initialize the environment

# Limb Environment
# from envs.limb_env import LimbEnv
# env = LimbEnv(render_mode=None)

# Quad Limb Environment
# from envs.quad_env import QuadEnv
# env = QuadEnv(render_mode=None)

# Rimless Wheel Environment
from envs.rimless_wheel_env import RimlessWheelEnv
env = RimlessWheelEnv(render_mode=None)


# 3. Define the Checkpoint Callback
# Save inside the run folder
checkpoint_callback = CheckpointCallback(
    save_freq=100_000, 
    save_path=f"{log_dir}/checkpoints", 
    name_prefix="limb_spin_model"
)

# 4. Initialize the Agent
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir
)

# 5. Train
print("Starting training...")
model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)

# 6. Save the final model INSIDE the log directory
final_path = os.path.join(log_dir, "limb_spin_final")
model.save(final_path)

print(f"Training finished. Final model saved to: {final_path}.zip")
print("To view model run:")
print(f"python play.py {final_path[2:]}")