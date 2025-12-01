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

        # --- THE NUCLEAR REWARD FUNCTION ---
        
        # 1. Spin Reward (THE ONLY THING THAT MATTERS)
        # We want it to rotate around the X-axis (Roll).
        # We pay huge points for high angular velocity.
        roll_velocity = self.data.qvel[2]
        reward_spin = 10.0 * abs(roll_velocity)

        # 2. Inversion Reward
        # Still useful to tell it WHICH direction to spin (upside down is good)
        reward_inverted = 5.0 * max(0.0, -1.0 * verticality)

        # 3. Energy Cost (Keep it small so it doesn't discourage effort)
        ctrl_cost = 1e-4 * np.sum(np.square(action))
   
        # 5. Handstand Bonus (Simplified)
        # Just getting inverted is the goal for now.
        reward_handstand = 0.0
        if verticality < -0.5:
            reward_handstand = 5.0

        # Sum it up
        reward = reward_spin + reward_inverted + reward_handstand - ctrl_cost

        # --- TERMINATION ---
        terminated = False
        # Lower the death threshold so it can struggle on the floor a bit
        if torso_z < 0.25: 
            terminated = True
            reward -= 10.0 # Small penalty. Failure is okay, inactivity is not.
            
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def reset_model(self, seed=None):
        noise_low = -0.05
        noise_high = 0.05
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        
        # --- THE FIX: RANDOMIZED START ---
        # 50% chance to spawn in a "Pre-Cartwheel" state
        if self.np_random.random() > 0.5:
            # Tilt sideways (0.3 rad)
            qpos[2] = 0.3 
            # Give it a shove (Velocity)
            qvel[0] = 1.5 # Moving left
            qvel[2] = 2.0 # Spinning left
            # Lift the leg (q3 is index 5)
            qpos[5] = 0.8 
        else:
            # Standard start
            qpos[0] = 0.0 
            qpos[1] = 0.0 
            qpos[2] = 0.0 

        self.set_state(qpos, qvel)
        return self._get_obs()