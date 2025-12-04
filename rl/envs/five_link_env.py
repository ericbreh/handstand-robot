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
        self._prev_roll_vel = 0.0
        self._prev_roll_angle = 0.0
        self._step_count = 0

    def _get_obs(self):
        obs_pos = self.data.qpos.flat[1:].copy() 
        velocity_data = self.data.qvel.flat.copy()
        sensor_data = self.data.sensordata.flat.copy() 
        return np.concatenate([obs_pos, velocity_data, sensor_data]).astype(np.float64)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        # --- EARLY NaN CHECK ---
        # Check if simulation exploded before using any values
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

        # Clip velocity to prevent explosion
        roll_angle = self.data.qpos[2]
        roll_vel = self.data.qvel[2]
        lateral_vel = self.data.qvel[1]
        
        # Track angular position change (want it to keep going in same direction)
        delta_angle = roll_angle - self._prev_roll_angle
        self._step_count += 1
        
        # Time elapsed: timestep (0.005) * frame_skip (5) * step_count
        elapsed_time = 0.005 * 5 * self._step_count
        
        reward = 0.0
        if delta_angle > 0:  # rotating in positive direction (forward progress)
            reward = 10.0 * abs(delta_angle)  # reward forward rotation
            reward = reward + 50.0 * abs(lateral_vel)
            

        else:  # went backwards
            reward = -100.0 * abs(delta_angle)  # heavy penalty for going backwards
        
        self._prev_roll_angle = roll_angle

        terminated = False
        
        # Termination conditions with reward/penalty based on rotation
        # roll_angle = abs(self.data.qpos[2])  # how much it has rotated
        
        if torso_z < 0.5:
            terminated = True
            if roll_angle > np.pi / 3:  # rotated significantly (~60 deg)
                reward += 50.0  # reduced from 100
                # True angular momentum: L = I × ω (from MuJoCo)
                roll_angular_momentum = self.data.subtree_angmom[0][0]  # X-axis (roll)
                reward += 100.0 * abs(roll_angular_momentum)
            else:  # fell without rotating - bad
                reward -= 100.0  # reduced from 100
        
        # Safety: flying away
        if torso_z > 5.0:
            terminated = True
        
        # Final safety clip on reward
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def reset_model(self, seed=None):
        noise_low = -0.05
        noise_high = 0.05
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        # qpos = self.init_qpos
        # qvel = self.init_qvel
        
        qpos[3] = 2.7
        qpos[4] = 2.7

        self.set_state(qpos, qvel)
        self._prev_roll_vel = 0.0
        self._prev_roll_angle = 0.0
        self._step_count = 0  # reset tracking
        return self._get_obs()