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

        # --- CORRECTED OBSERVATION SPACE CALCULATION ---
        # 1. qpos (excluding x,y): 11 total - 2 = 9
        # 2. qvel: 10
        # 3. Sensors:
        #    - 4x joint pos
        #    - 4x joint vel
        #    - 1x framequat (4 values)
        #    - 1x framelinvel (3 values)
        #    - 4x touch sensors
        #    Total sensor size = 4 + 4 + 4 + 3 + 4 = 19
        # TOTAL OBS SIZE = 9 + 10 + 19 = 38
        observation_space = Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float64)

        super().__init__(
            model_path=MODEL_PATH,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs,
        )

    def _get_obs(self):
        # 1. Position Data (11 elements)
        position_data = self.data.qpos.flat.copy()
        # We skip indices 0 and 1 (Global X and Y) so the agent acts the same anywhere on the floor
        # We keep Z (height), Quat (orientation), and Joint Angles
        obs_pos = position_data[2:] # Size 9

        # 2. Velocity Data (10 elements)
        velocity_data = self.data.qvel.flat.copy() # Size 10

        # 3. Sensor Data (19 elements)
        # Includes interleaved joint pos/vel, torso info, and TOUCH sensors
        sensor_data = self.data.sensordata.flat.copy() # Size 19

        return np.concatenate([obs_pos, velocity_data, sensor_data]).astype(np.float64)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        # --- EXTRACT INFO ---
        # Height of torso center
        torso_z = self.data.qpos[2]
        
        # Orientation (Z-axis of the body)
        # We want this to be -1 (pointing down) for a handstand
        rot_matrix = self.data.body("torso").xmat.reshape(3, 3)
        verticality = rot_matrix[2, 2] # Global Z component of Body Z axis

        # Touch Sensors (Last 4 elements: R_Hand, L_Hand, R_Foot, L_Foot)
        touch_sensors = self.data.sensordata[-4:] 
        hand_contact = np.sum(touch_sensors[0:2])
        foot_contact = np.sum(touch_sensors[2:4])
        
        # --- REWARD FUNCTION ---
        
        # We want the robot to move forward (positive X) to generate momentum.
        # This helps it kick over rather than just trying to press up.
        x_velocity = self.data.qvel[0]
        reward_velocity = 1.0 * x_velocity

        # 2. Inversion Reward
        # Reward being upside down (verticality near -1)
        reward_inverted = -1.0 * verticality 
        
        # 3. Handstand Bonus
        # Big bonus if inverted AND touching ground with hands
        reward_handstand = 0.0
        if verticality < -0.8 and hand_contact > 1.0:
            reward_handstand = 5.0
            
            # Bonus for stillness (holding the pose)
            # This balances out the velocity reward so it learns to STOP after flipping
            velocity_magnitude = np.linalg.norm(self.data.qvel)
            if velocity_magnitude < 1.0:
                reward_handstand += 5.0

        # 4. Foot Contact Penalty
        # If we are inverted but our feet touch, we failed the balance
        reward_feet = 0.0
        if verticality < 0 and foot_contact > 0.1:
            reward_feet = -5.0

        # 5. Control Cost (Efficiency)
        ctrl_cost = 1e-3 * np.sum(np.square(action))

        # Sum it all up
        reward = reward_velocity + reward_inverted + reward_handstand + reward_feet - ctrl_cost

        # --- TERMINATION ---
        terminated = False
        
        # Fail: Torso too low (crashed)
        if torso_z < 0.4: 
            terminated = True
            reward -= 10.0
            
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def reset_model(self, seed=None):
        noise_low = -0.05
        noise_high = 0.05

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        
        # Force start height and orientation
        qpos[2] = 1.25 
        qpos[3] = 1.0 # w
        qpos[4:7] = 0.0 # x, y, z

        self.set_state(qpos, qvel)
        return self._get_obs()