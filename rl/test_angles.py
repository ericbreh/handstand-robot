import gymnasium as gym
import numpy as np
from envs.five_link_env import FiveLinkCartwheelEnv

# Create env with render mode human to see it
env = FiveLinkCartwheelEnv(render_mode="human")
env.reset()

print("--- ANGLE TEST ---")
print("RIGHT ARM (Green) set to 0.0")
print("LEFT ARM (Pink) set to 1.5 (approx 90 deg)")

for _ in range(1000):
    # Override the physics state manually
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()

    # 1. LOCK ROTATION (So it doesn't fall over and distract you)
    qpos[0] = 0 # Y
    qpos[1] = 0 # Z
    qpos[2] = 3.14 # Roll (Upright)

    # 2. SET TEST ANGLES
    qpos[3] = 1.5  # Right Arm (Test 0)
    qpos[4] = 1.5  # Left Arm (Test 1.5)

    # Apply state
    env.set_state(qpos, qvel)
    
    # Render
    env.render()