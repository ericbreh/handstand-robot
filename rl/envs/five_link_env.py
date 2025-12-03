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

        #  !! REWARD SHAPING !!

        # 1. Spin Reward (THE ONLY THING THAT MATTERS)
        roll_velocity = self.data.qvel[2]
        reward_spin = 10.0 * roll_velocity

        # 2. Energy Cost)
        # ctrl_cost = 1e-4 * np.sum(np.square(action))
   
        # Sum it up
        reward = reward_spin
        
        # --- TERMINATION ---
        terminated = False
        # Lower the death threshold so it can struggle on the floor a bit
        if torso_z < 0.25: 
            terminated = True
            reward -= 10.0 # Small penalty. Failure is okay, inactivity is not.

        return self._get_obs(), reward, terminated, False, {}

    def reset_model(self, seed=None):
        qpos = self.init_qpos
        qvel = self.init_qvel
        
        self.set_state(qpos, qvel)
        return self._get_obs()