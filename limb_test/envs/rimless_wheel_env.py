import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class RimlessWheelEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/rimless_wheel_model.xml"))
        
        # --- CRITICAL FIX: OBSERVATION SHAPE ---
        # Old Shape: 14. 
        # New Shape: 15 (We removed raw theta (1) and added sin+cos (2))
        # 14 - 1 + 2 = 15
        observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)
        
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=1,
            observation_space=observation_space,
            render_mode=render_mode
        )

        self.max_steps = 10000
        self.current_step = 0
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.current_step += 1

        ob = self._get_obs()
        
        # --- 1. SPEED REWARD ---
        # qvel[2] is the Angular Velocity of the Hub.
        angular_velocity = self.data.qvel[2]
        
        # Reward: Higher is better.
        speed_reward = 4.0 * angular_velocity 
        

        # --- REMOVED SYNC PENALTY ---
        # We want the robot to kick! Penalizing difference in leg speed prevents kicking.
        
        reward = speed_reward

        # --- TERMINATION CHECK ---
        # Real Height = qpos[1] + 0.6
        real_height = self.data.qpos[1] + 0.6
        
        terminated = False
        # If hub drops below 0.25m, it crashed.
        if real_height < 0.25:
            reward -= 100.0 # Big penalty for crashing
            terminated = True 

        # --- TIMEOUT CHECK ---
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, truncated, {}

    def _get_obs(self):
        # --- CRITICAL FIX: CYCLIC OBSERVATION ---
        # The raw angle qpos[2] grows to infinity (0, 10, 100, 1000...).
        # This confuses the Neural Network.
        # We replace "Angle" with "Sin(Angle)" and "Cos(Angle)".
        
        qpos = self.data.qpos.flat[:]
        qvel = self.data.qvel.flat[:]
        
        theta = qpos[2] # The Hub Rotation
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Exclude the raw theta (index 2)
        # qpos is: [x, z, theta, leg1, leg2, leg3, leg4]
        # We want: [x, z, leg1, leg2, leg3, leg4] + [sin, cos]
        
        new_qpos = np.delete(qpos, 2) 
        
        return np.concatenate([new_qpos, qvel, [sin_theta, cos_theta]])

    def reset_model(self):
        self.current_step = 0

        # Add noise to initial state
        qpos = self.init_qpos + self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nv)
        
        # Fix 1: Place on Ground
        qpos[1] = -0.1 

        # Fix 2: Initial Push (Momentum)
        qvel[2] = 15.0  
        
        # Reset Legs to neutral
        qpos[3:] = 0
        qvel[3:] = 0
        
        qpos[0] = 0
        
        self.set_state(qpos, qvel)
        return self._get_obs()