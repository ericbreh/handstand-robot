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
        obs_pos = self.data.qpos.flat.copy()
        velocity_data = self.data.qvel.flat.copy()
        sensor_data = self.data.sensordata.flat.copy()
        return np.concatenate([obs_pos, velocity_data, sensor_data]).astype(np.float64)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        # --- Extract sensor and state data ---
        torso_z_position = self.data.body("torso").xpos[2]

        right_foot_contact = self.data.sensor("touch_right_foot").data[0] > 0
        left_foot_contact = self.data.sensor("touch_left_foot").data[0] > 0

        # --- Curriculum Task 1: Balance on the Right Leg (Corrected) ---

        reward = 0.0  # Initialize to neutral

        # If goal achieved, give large positive reward
        if right_foot_contact and not left_foot_contact:
            reward = 2.0
        # If left foot is on ground (either alone or with right), apply penalty
        elif left_foot_contact:
            reward = -1.5
        #
        # # Stability Penalty: Encourage smooth, minimal movement
        # stability_cost = -0.01 * (
        #     np.sum(np.square(self.data.qvel)) + np.sum(np.square(action))
        # )
        # reward += stability_cost # Add stability cost to the main reward
        #
        # --- Termination Conditions ---
        # Terminate if the torso falls too low
        terminated = torso_z_position < 0.7
        if terminated:
            reward = -20.0  # Heavy penalty for falling

        return self._get_obs(), reward, terminated, False, {}

    def reset_model(self):
        """
        Resets the model to a randomly perturbed initial state.
        This is crucial for training a robust policy.
        """
        # Define the magnitude of the random noise
        pos_noise_magnitude = 0.02
        vel_noise_magnitude = 0.1

        # Generate random noise centered around zero
        # self.np_random is seeded by the parent's reset() method
        qpos_noise = self.np_random.uniform(
            low=-pos_noise_magnitude,
            high=pos_noise_magnitude,
            size=self.init_qpos.shape,
        )
        qvel_noise = self.np_random.uniform(
            low=-vel_noise_magnitude,
            high=vel_noise_magnitude,
            size=self.init_qvel.shape,
        )

        # Combine initial state with noise
        noisy_qpos = self.init_qpos + qpos_noise
        noisy_qvel = self.init_qvel + qvel_noise

        # Set the new, noisy initial state
        self.set_state(noisy_qpos, noisy_qvel)

        return self._get_obs()
