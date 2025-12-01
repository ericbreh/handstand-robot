import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from os import path

MODEL_PATH = path.join(path.dirname(__file__), '..', 'models', 'robot_model.xml')

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

    def _get_obs(self):
        obs_pos = self.data.qpos.flat[1:].copy() 
        velocity_data = self.data.qvel.flat.copy()
        sensor_data = self.data.sensordata.flat.copy() 
        return np.concatenate([obs_pos, velocity_data, sensor_data]).astype(np.float64)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        # --- EXTRACT INFO ---
        torso_z = self.data.body("torso").xpos[2] 
        rot_matrix = self.data.body("torso").xmat.reshape(3, 3)
        verticality = rot_matrix[2, 2] 

        touch_sensors = self.data.sensordata[-4:] 
        hand_contact = np.sum(touch_sensors[0:2])
        foot_contact = np.sum(touch_sensors[2:4])
        
        # Geometry Info
        right_foot_z = self.data.site("right_foot_site").xpos[2]
        left_foot_z = self.data.site("left_foot_site").xpos[2]
        avg_feet_height = (right_foot_z + left_foot_z) / 2.0
        
        right_hand_z = self.data.site("right_hand_site").xpos[2]
        left_hand_z = self.data.site("left_hand_site").xpos[2]
        min_hand_height = min(right_hand_z, left_hand_z)

        # --- REWARD FUNCTION ---
        
        # 1. Alive Bonus
        reward_alive = 0.2

        # 2. Velocity Reward
        y_velocity = self.data.qvel[0] 
        reward_velocity = 2.0 * abs(y_velocity) 

        # 3. Spin Reward
        roll_velocity = self.data.qvel[2]
        # Base spin reward
        reward_spin = 3.0 * abs(roll_velocity)

        # 4. Inversion Guide
        reward_potential = 5.0 * (1.0 - verticality)

        # 5. Feet Lift Reward
        reward_feet_lift = 10.0 * avg_feet_height
        
        # 6. DIVE REWARD (The Smart Fix)
        # Reward lowering hands ONLY if we are spinning.
        # This prevents "squatting" for points.
        # 1.0 - min_hand_height is big when hands are low.
        # roll_velocity is big when flipping.
        # Product is huge when doing a cartwheel entry.
        reward_dive = 5.0 * (1.0 - min_hand_height) * abs(roll_velocity)

        # 7. Flight Bonus
        reward_flight = 0.0
        if hand_contact > 0.1 and foot_contact < 0.1:
             reward_flight = 20.0 

        # 8. Handstand Bonus
        reward_handstand = 0.0
        if verticality < -0.8 and hand_contact > 1.0:
            reward_handstand = 20.0 
            if np.linalg.norm(self.data.qvel) < 2.0:
                reward_handstand += 30.0

        # 9. Penalties
        reward_feet_penalty = 0.0
        if verticality < 0 and foot_contact > 0.1:
            reward_feet_penalty = -5.0

        ctrl_cost = 1e-3 * np.sum(np.square(action))

        reward = (reward_alive + 
                  reward_velocity + 
                  reward_spin + 
                  reward_potential + 
                  reward_feet_lift + 
                  reward_dive + # Added the smart dive reward
                  reward_flight + 
                  reward_handstand + 
                  reward_feet_penalty - 
                  ctrl_cost)

        # --- TERMINATION ---
        terminated = False
        if torso_z < 0.3: 
            terminated = True
            reward -= 10.0 
            
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def reset_model(self, seed=None):
        noise_low = -0.05
        noise_high = 0.05
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        
        qpos[0] = 0.0 
        qpos[1] = 0.0 
        qpos[2] = 0.0 

        self.set_state(qpos, qvel)
        return self._get_obs()