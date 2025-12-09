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
        self._stage = 'swing_up'

        self._persist_data = {
            'prev_roll_angle': 0.0,
            'prev_roll_vel': 0.0,
            'prev_lateral_vel': 0.0,
            'prev_right_foot_touch': 0.0,
            'prev_left_foot_touch': 0.0,
            'step_count': 0,
        }

    def _get_obs(self):
        obs_pos = self.data.qpos.flat[1:].copy() 
        velocity_data = self.data.qvel.flat.copy()
        sensor_data = self.data.sensordata.flat.copy() 
        return np.concatenate([obs_pos, velocity_data, sensor_data]).astype(np.float64)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        

        terminated = False

        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            obs = np.zeros(25, dtype=np.float64)
            return obs, 0.0, True, False, {}
        


        # --- EXTRACT INFO ---
        torso_z = self.data.body("torso").xpos[2] 
        rot_matrix = self.data.body("torso").xmat.reshape(3, 3)
        verticality = rot_matrix[2, 2] 
        com_position = self.data.subtree_com[self.model.body("world").id]

        right_foot_pos = self.data.site("right_foot_site").xpos
        left_foot_pos = self.data.site("left_foot_site").xpos

        # From your sensor data (gives scalar contact values)
        right_foot_touch = self.data.sensor("touch_right_foot").data[0]
        left_foot_touch = self.data.sensor("touch_left_foot").data[0]

        # Clip velocity to prevent explosion
        roll_angle = self.data.qpos[2]
        roll_vel = self.data.qvel[2]
        lateral_vel = self.data.qvel[1]
        
        delta_angle = roll_angle - self._persist_data['prev_roll_angle']

        reward = []

        if self._stage == 'swing_up': # Trying to get maximum angular momentum here
            # Track angular position change (want it to keep going in same direction)
            
            if delta_angle > 0:  # rotating in positive direction (forward progress)
                reward.append(10.0 * abs(delta_angle))  # reward forward rotation
                reward.append(50.0 * abs(lateral_vel))
            else:  # went backwards
                reward.append(-100.0 * abs(delta_angle))  # heavy penalty for going backwards
            
            if abs(roll_angle) > np.pi / 2:
                if roll_angle > 0:
                    reward.append(50.0)  # reduced from 100
                    # True angular momentum: L = I × ω (from MuJoCo)
                    roll_angular_momentum = self.data.subtree_angmom[0][0]  # X-axis (roll)
                    reward.append(150.0 * abs(roll_angular_momentum))
                    terminated = True
                else:
                    reward.append(-100.0)  # reduced from 100
                    terminated = True

        elif self._stage == 'landing':
            pass

        if torso_z < 0.5:
            reward.append(-100.0)
            terminated = True
        #     if roll_angle > np.pi / 3:  # rotated significantly (~60 deg)
                
        #     else:  # fell without rotating - bad
        #         reward.append(-100.0)  # reduced from 100
        
        # Safety: flying away
        if torso_z > 5.0:
            terminated = True

        # Update persistent data
        self._persist_data['prev_roll_angle'] = roll_angle

        
        # Final safety clip on reward
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), sum(reward), terminated, False, {}

    def reset_model(self, seed=None):
        noise_low = -0.05
        noise_high = 0.05
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        
        qpos[3] = 2.7
        qpos[4] = 2.7

        self.set_state(qpos, qvel)
        self._prev_roll_vel = 0.0
        self._prev_roll_angle = 0.0
        self._step_count = 0  # reset tracking
        return self._get_obs()