import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from os import path
import mujoco

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
        self._step_count = 0

    def _get_obs(self):
        obs_pos = self.data.qpos.flat[1:].copy()
        velocity_data = self.data.qvel.flat.copy()
        sensor_data = self.data.sensordata.flat.copy()
        return np.concatenate([obs_pos, velocity_data, sensor_data]).astype(np.float64)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        # Check if simulation exploded
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            obs = np.zeros(25, dtype=np.float64)
            return obs, 0.0, True, False, {}

        self._step_count += 1
        info = {}

        torso_z = self.data.body("torso").xpos[2]
        rot_matrix = self.data.body("torso").xmat.reshape(3, 3)
        verticality = rot_matrix[2, 2]

        reward = 0.0

        # HAND CONTACT
        left_hand_on_ground = False
        right_hand_on_ground = False
        ground_geom_id = self.model.geom("ground").id
        left_arm_geom_id = self.model.geom("left_arm_geom").id
        right_arm_geom_id = self.model.geom("right_arm_geom").id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (
                contact.geom1 == left_arm_geom_id and contact.geom2 == ground_geom_id
            ) or (
                contact.geom1 == ground_geom_id and contact.geom2 == left_arm_geom_id
            ):
                left_hand_on_ground = True
            if (
                contact.geom1 == right_arm_geom_id and contact.geom2 == ground_geom_id
            ) or (
                contact.geom1 == ground_geom_id and contact.geom2 == right_arm_geom_id
            ):
                right_hand_on_ground = True

        hand_contact_reward = 0.0
        if left_hand_on_ground and right_hand_on_ground:
            hand_contact_reward += 100.0
        info["hand_contact_reward"] = hand_contact_reward
        reward += hand_contact_reward

        # LEG CONTACT
        left_foot_on_ground = False
        right_foot_on_ground = False
        ground_geom_id = self.model.geom("ground").id
        left_leg_geom_id = self.model.geom("left_leg_geom").id
        right_leg_geom_id = self.model.geom("right_leg_geom").id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (
                contact.geom1 == left_leg_geom_id and contact.geom2 == ground_geom_id
            ) or (
                contact.geom1 == ground_geom_id and contact.geom2 == left_leg_geom_id
            ):
                left_foot_on_ground = True
            if (
                contact.geom1 == right_leg_geom_id and contact.geom2 == ground_geom_id
            ) or (
                contact.geom1 == ground_geom_id and contact.geom2 == right_leg_geom_id
            ):
                right_foot_on_ground = True

        # ONE HAND ONE FOOT PENALTY
        one_hand_one_foot_penalty = 0.0
        if (
            (left_hand_on_ground or right_hand_on_ground)
            and (left_foot_on_ground or right_foot_on_ground)
            and not (left_hand_on_ground and right_hand_on_ground)
            and not (left_foot_on_ground and right_foot_on_ground)
        ):
            one_hand_one_foot_penalty = -50.0
        info["one_hand_one_foot_penalty"] = one_hand_one_foot_penalty
        reward += one_hand_one_foot_penalty

        # SUSTAINED BALANCE REWARD
        sustained_balance_reward = 0.0
        if (left_hand_on_ground and right_hand_on_ground) and (
            not left_foot_on_ground and not right_foot_on_ground
        ):
            sustained_balance_reward = 100.0
        info["sustained_balance_reward"] = sustained_balance_reward
        reward += sustained_balance_reward

        # INVERSION REWARD
        inversion_reward = 100.0 * (1.0 - np.sqrt(np.maximum(verticality + 1.0, 0.0)))
        info["inversion_reward"] = inversion_reward
        reward += inversion_reward

        terminated = False
        termination_penalty = 0.0
        if torso_z < 0.5:
            terminated = True
            termination_penalty = -200.0
        info["termination_penalty"] = termination_penalty
        reward += termination_penalty

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, info

    def reset_model(self, seed=None):
        noise_low = -0.05
        noise_high = 0.05
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        qpos[3] = 2.7
        qpos[4] = 2.7
        self.set_state(qpos, qvel)
        self._step_count = 0
        return self._get_obs()
