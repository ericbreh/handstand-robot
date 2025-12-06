import gymnasium as gym

# Initialize the custom environment
# Limb Environment
# from envs.limb_env import LimbEnv
# env = LimbEnv(render_mode="human")

# Quad Limb Environment
# from envs.quad_env import QuadEnv
# env = QuadEnv(render_mode="human")

# Rimless Wheel Environment
from envs.rimless_wheel_env import RimlessWheelEnv
env = RimlessWheelEnv(render_mode="human")


observation, info = env.reset()

print("Visualizing model... Press Ctrl+C to stop.")

# Run an infinite loop with random actions just to see physics
try:
    while True:
        # Sample a random action
        action = env.action_space.sample() 
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
except KeyboardInterrupt:
    print("Stopped.")
    env.close()