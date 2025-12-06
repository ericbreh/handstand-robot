import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class QuadEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/quad_model.xml"))
        
        # CHANGED: Shape is now 10 (5 joints * 2 values)
        # Joints: [Hub, Limb1, Limb2, Limb3, Limb4]
        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=1,
            observation_space=observation_space,
            render_mode=render_mode
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        ob = self._get_obs()
        
        # CHANGED: Reward is velocity of the HUB joint.
        # The obs vector is [qpos(5), qvel(5)]
        # Index 5 is the first velocity item -> The Hub Velocity
        hub_velocity = ob[5] 
        reward = hub_velocity

        terminated = False
        truncated = False
        
        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()