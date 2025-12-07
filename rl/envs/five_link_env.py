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
        observation_space = Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64)
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

        # --- EXTRACT INFO ---
        torso_z = self.data.body("torso").xpos[2]
        rot_matrix = self.data.body("torso").xmat.reshape(3, 3)
        verticality = rot_matrix[2, 2]

        com_position = self.data.subtree_com[self.model.body("world").id]

        left_hand_contact = self.data.sensordata[8]
        right_hand_contact = self.data.sensordata[9]
        left_foot_contact = self.data.sensordata[10]
        right_foot_contact = self.data.sensordata[11]

        reward = 0.0

        # ARMS UP REWARD
        right_shoulder_angle = self.data.qpos[3]
        left_shoulder_angle = self.data.qpos[4]
        arms_up_reward = 0.0
        max_shoulder_angle = 3.0  # Based on user's input: arms at sides is ~3 radians

        # Reward for arms being straight up (angle close to 0)
        # We use max_shoulder_angle - abs(angle) to give max reward at 0 and min at max_shoulder_angle
        arms_up_reward += (
            10.0 * (max_shoulder_angle - abs(left_shoulder_angle)) / max_shoulder_angle
        )
        arms_up_reward += (
            10.0 * (max_shoulder_angle - abs(right_shoulder_angle)) / max_shoulder_angle
        )

        # Ensure reward is not negative if angle goes beyond max_shoulder_angle
        arms_up_reward = max(0.0, arms_up_reward)

        reward += arms_up_reward

        # HAND CONTACT REWARD
        left_hand_on_ground = False
        right_hand_on_ground = False

        ground_geom_id = self.model.geom("ground").id
        left_arm_geom_id = self.model.geom("left_arm_geom").id
        right_arm_geom_id = self.model.geom("right_arm_geom").id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Check for left hand contact with ground
            if (
                contact.geom1 == left_arm_geom_id and contact.geom2 == ground_geom_id
            ) or (
                contact.geom1 == ground_geom_id and contact.geom2 == left_arm_geom_id
            ):
                left_hand_on_ground = True
            # Check for right hand contact with ground
            if (
                contact.geom1 == right_arm_geom_id and contact.geom2 == ground_geom_id
            ) or (
                contact.geom1 == ground_geom_id and contact.geom2 == right_arm_geom_id
            ):
                right_hand_on_ground = True

        hand_contact_reward = 0.0
        if left_hand_on_ground:
            hand_contact_reward += 20.0
        if right_hand_on_ground:
            hand_contact_reward += 20.0

        if left_hand_on_ground and right_hand_on_ground:
            hand_contact_reward += 100.0  # Large bonus reward

        reward += hand_contact_reward

        # INVERSION REWARD
        inversion_reward = 100.0 * (1.0 - (verticality + 1.0) ** 2)
        reward += inversion_reward

        # LEG CONTACT PENALTY
        left_foot_on_ground = False
        right_foot_on_ground = False

        ground_geom_id = self.model.geom("ground").id
        left_leg_geom_id = self.model.geom("left_leg_geom").id
        right_leg_geom_id = self.model.geom("right_leg_geom").id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Check for left foot contact with ground
            if (
                contact.geom1 == left_leg_geom_id and contact.geom2 == ground_geom_id
            ) or (
                contact.geom1 == ground_geom_id and contact.geom2 == left_leg_geom_id
            ):
                left_foot_on_ground = True
            # Check for right foot contact with ground
            if (
                contact.geom1 == right_leg_geom_id and contact.geom2 == ground_geom_id
            ) or (
                contact.geom1 == ground_geom_id and contact.geom2 == right_leg_geom_id
            ):
                right_foot_on_ground = True

        leg_contact_penalty = 0.0
        if left_foot_on_ground:
            leg_contact_penalty -= 50.0
        if right_foot_on_ground:
            leg_contact_penalty -= 50.0
        reward += leg_contact_penalty

        terminated = False
        if torso_z < 0.5:
            terminated = True
            reward -= 200.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

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
