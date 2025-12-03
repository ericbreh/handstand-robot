import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from os import path

MODEL_PATH = path.join(path.dirname(__file__), "..", "models", "robot_model.xml")


class FiveLinkCartwheelEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(self, **kwargs):
        gym.utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float64)
        super().__init__(
            model_path=MODEL_PATH,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs,
        )

    def _get_obs(self):
        obs_pos = (
            self.data.qpos.flat.copy()
        )  # <-- FIXED: Was skipping the first element (y-position)
        velocity_data = self.data.qvel.flat.copy()
        sensor_data = self.data.sensordata.flat.copy()
        return np.concatenate([obs_pos, velocity_data, sensor_data]).astype(np.float64)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        # --- EXTRACT INFO ---
        torso_z = self.data.body("torso").xpos[2]

        #  !! REWARD SHAPING !!

        # Correctly identify forward and roll velocities
        forward_velocity = self.data.qvel[
            0
        ]  # <-- FIXED: Was using qvel[1] (vertical) instead of qvel[0] (forward)
        roll_velocity = self.data.qvel[2]

        # A more robust reward for continuous cartwheeling
        cartwheel_reward = (0.5 * forward_velocity) + (0.5 * abs(roll_velocity))

        # Add a small penalty for every timestep to encourage action
        alive_penalty = -0.1

        # Add a small reward for taking large actions (encourages using motors)
        action_reward = 0.01 * np.mean(np.abs(action))

        reward = cartwheel_reward + alive_penalty + action_reward

        # --- TERMINATION ---
        terminated = False
        # The agent fails if it falls over
        if torso_z < 0.25:
            terminated = True
            reward = -10.0  # Give a flat, large penalty for falling.

        return self._get_obs(), reward, terminated, False, {}

    def reset_model(self, seed=None):
        qpos = self.init_qpos
        qvel = self.init_qvel

        self.set_state(qpos, qvel)
        return self._get_obs()
