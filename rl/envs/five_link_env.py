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
        
        # Get Feet Heights (To break the "stretching" habit)
        right_foot_z = self.data.site("right_foot_site").xpos[2]
        left_foot_z = self.data.site("left_foot_site").xpos[2]
        avg_feet_height = (right_foot_z + left_foot_z) / 2.0

        # --- REWARD FUNCTION ---
        
        # 1. Alive Bonus
        reward_alive = 0.5

        # 2. Velocity Reward (Momentum)
        y_velocity = self.data.qvel[0] 
        reward_velocity = 2.0 * abs(y_velocity) 

        # 3. Spin Reward (Rotation)
        roll_velocity = self.data.qvel[2]
        reward_spin = 3.0 * abs(roll_velocity)

        # 4. Inversion Guide
        reward_potential = 7.0 * (1.0 - verticality)

        # 5. Feet Height Reward (The key fix for stretching)
        # 0.0 (ground) -> 0 pts. 1.9m (inverted) -> 19 pts.
        reward_feet_lift = 15.0 * avg_feet_height
        
        # 6. Flight Bonus
        # Hands down + Feet up = Huge Bonus
        reward_flight = 0.0
        if hand_contact > 0.1 and foot_contact < 0.1:
             reward_flight = 20.0 

        # 7. Handstand Bonus
        reward_handstand = 0.0
        if verticality < -0.8 and hand_contact > 1.0:
            reward_handstand = 20.0 
            if np.linalg.norm(self.data.qvel) < 2.0:
                reward_handstand += 30.0

        # 8. Penalties
        reward_feet_penalty = 0.0
        if verticality < 0 and foot_contact > 0.1:
            reward_feet_penalty = -5.0 

        ctrl_cost = 1e-3 * np.sum(np.square(action))

        reward = (reward_alive + 
                  reward_velocity + 
                  reward_spin + 
                  reward_potential + 
                  reward_feet_lift + 
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