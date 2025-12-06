import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class LimbEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        # Get absolute path to your XML
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/limb_model.xml"))
        
        # Frame skip = 1 means we calculate physics every step (high precision)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=1,
            observation_space=observation_space,
            render_mode=render_mode
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        # 1. Apply the action (torque)
        self.do_simulation(action, self.frame_skip)
        
        # 2. Get observations (position and velocity)
        ob = self._get_obs()
        
        # 3. Calculate Reward: Simple! Reward = Angular Velocity
        # We want it to spin fast.
        velocity = ob[1]
        reward = velocity 

        # 4. Check termination (never ends, it just spins)
        terminated = False
        truncated = False
        
        # 5. Render frame if needed
        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, truncated, {}

    def _get_obs(self):
        # Observation: [Joint Angle, Joint Velocity]
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def reset_model(self):
        # Reset to a random position slightly off-center
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()