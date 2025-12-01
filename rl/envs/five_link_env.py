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
        
        # --- REWARD FUNCTION (AGGRESSIVE UPDATE) ---
        
        # 1. Alive Bonus (SLASHED)
        # Standing still gets you 500 pts total. Not enough to be satisfied.
        # But still better than crashing (-100).
        reward_alive = 0.5

        # 2. Velocity Reward (BOOSTED)
        # Increased from 0.5 to 5.0. 
        # Moving sideways is now the MAIN source of income for the first phase.
        # It forces the robot to step/lean/run sideways.
        y_velocity = self.data.qvel[0] 
        reward_velocity = 5.0 * abs(y_velocity) 

        # 3. Inversion Reward (The Jackpot)
        reward_inverted = 10.0 * max(0.0, -1.0 * verticality)
        
        # 4. Handstand Bonus
        reward_handstand = 0.0
        if verticality < -0.8 and hand_contact > 1.0:
            reward_handstand = 5.0
            if np.linalg.norm(self.data.qvel) < 2.0:
                reward_handstand += 5.0

        # 6. Foot Contact Penalty
        reward_feet = 0.0
        if verticality < 0 and foot_contact > 0.1:
            reward_feet = -5.0

        ctrl_cost = 1e-3 * np.sum(np.square(action))

        # Sum it all up
        reward = (reward_alive + 
                  reward_velocity + 
                  reward_inverted + 
                  reward_handstand + 
                  reward_feet - 
                  ctrl_cost)

        # --- TERMINATION ---
        terminated = False
        if torso_z < 0.3: 
            terminated = True
            reward -= 100.0
            
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