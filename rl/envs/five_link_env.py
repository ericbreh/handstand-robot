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
        info = {}

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

        # # ARMS UP REWARD
        # right_shoulder_angle = self.data.qpos[3]
        # left_shoulder_angle = self.data.qpos[4]
        # arms_up_reward = 0.0
        # max_shoulder_angle = 2.79
        #
        # # Reward for arms being straight up (angle close to 0)
        # # We use max_shoulder_angle - abs(angle) to give max reward at 0 and min at max_shoulder_angle
        # arms_up_reward += (
        #     10.0 * (max_shoulder_angle - abs(left_shoulder_angle)) / max_shoulder_angle
        # )
        # arms_up_reward += (
        #     10.0 * (max_shoulder_angle - abs(right_shoulder_angle)) / max_shoulder_angle
        # )
        # arms_up_reward = max(0.0, arms_up_reward)
        # info["arms_up_reward"] = arms_up_reward
        # reward += arms_up_reward

        # # LEGS STRAIGHT REWARD
        # right_hip_angle = self.data.qpos[5]
        # left_hip_angle = self.data.qpos[6]
        # legs_straight_reward = 0.0
        # max_hip_angle = 1.59
        #
        # # Reward for legs being straight (angle close to 0)
        # legs_straight_reward += (
        #     10.0 * (max_hip_angle - abs(left_hip_angle)) / max_hip_angle
        # )
        # legs_straight_reward += (
        #     10.0 * (max_hip_angle - abs(right_hip_angle)) / max_hip_angle
        # )
        # legs_straight_reward = max(0.0, legs_straight_reward)
        # info["legs_straight_reward"] = legs_straight_reward
        # reward += legs_straight_reward

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
        # if left_hand_on_ground:
        #     hand_contact_reward += 20.0
        # if right_hand_on_ground:
        #     hand_contact_reward += 20.0

        if left_hand_on_ground and right_hand_on_ground:
            hand_contact_reward += 50.0

        info["hand_contact_reward"] = hand_contact_reward
        reward += hand_contact_reward

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
        # if left_foot_on_ground:
        #     leg_contact_penalty -= 50.0
        # if right_foot_on_ground:
        #     leg_contact_penalty -= 50.0

        if (not left_foot_on_ground) and (not right_foot_on_ground):
            leg_contact_penalty += 50
        info["leg_contact_penalty"] = leg_contact_penalty
        reward += leg_contact_penalty

        # # ONE HAND ONE FOOT PENALTY
        # one_hand_one_foot_penalty = 0.0
        # if (
        #     (left_hand_on_ground or right_hand_on_ground)
        #     and (left_foot_on_ground or right_foot_on_ground)
        #     and not (left_hand_on_ground and right_hand_on_ground)
        #     and not (left_foot_on_ground and right_foot_on_ground)
        # ):
        #     one_hand_one_foot_penalty = -50.0
        # info["one_hand_one_foot_penalty"] = one_hand_one_foot_penalty
        # reward += one_hand_one_foot_penalty

        # INVERSION REWARD
        inversion_reward = 100.0 * (1.0 - np.sqrt(np.maximum(verticality + 1.0, 0.0)))
        info["inversion_reward"] = inversion_reward
        reward += inversion_reward

        terminated = False
        if (
            torso_z
            < 0.5
            # or abs(right_hip_angle) >= 1.45
            # and abs(left_hip_angle) >= 1.45
        ):
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
