import gymnasium as gym
import time

env = gym.make("Walker2d-v5", render_mode="human")

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

obs, info = env.reset(seed=42)

for _ in range(1000):
    # Take a random action
    action = env.action_space.sample()

    # Get the results of that action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished. Resetting.")
        obs, info = env.reset()

    time.sleep(0.01)

env.close()
