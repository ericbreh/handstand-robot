import sys
import mujoco 
from stable_baselines3 import PPO

# 1. Load environment

# Limb Environment
# from envs.limb_env import LimbEnv
# env = LimbEnv(render_mode="human")

# Quad Limb Environment
# from envs.quad_env import QuadEnv
# env = QuadEnv(render_mode="human")

# Rimless Wheel Environment
from envs.rimless_wheel_env import RimlessWheelEnv
env = RimlessWheelEnv(render_mode="human")

# 2. Load model
model_path = sys.argv[1]
model = PPO.load(model_path)
print(f"Loaded model from: {model_path}")

# 3. Run simulation
obs, _ = env.reset()
print("Playing! Press Ctrl+C to stop.")

try:
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        # Print reward in Mujoco viewer
        renderer = env.unwrapped.mujoco_renderer
        if renderer.viewer:
            # Add text to Top-Left of the screen
            renderer.viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                "Total Reward:", 
                f"{reward:.4f}"
            )

        if done or truncated:
            obs, _ = env.reset()

except KeyboardInterrupt:
    env.close()