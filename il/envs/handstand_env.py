import gymnasium as gym
import numpy as np
import mujoco
import os
from gymnasium import spaces

class HandstandEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Load Model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "../models/recorder_model.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Load Expert Trajectory
        traj_path = os.path.join(current_dir, "../expert_trajectory_full.npy")
        if not os.path.exists(traj_path):
            raise FileNotFoundError("Run interpolate_trajectory.py first!")
        
        self.expert_data = np.load(traj_path, allow_pickle=True).item()
        self.expert_qpos = self.expert_data["qpos"]
        self.expert_qvel = self.expert_data["qvel"]
        
        self.trajectory_dt = self.expert_data.get("dt", 0.002)
        self.max_steps = len(self.expert_qpos)
        
        # Cache the Ground ID to distinguish floor contacts from self-collisions
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        if self.ground_id == -1:
             self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Define Action Space
        n_actions = self.model.nu
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()
        
        # Define Observation Space
        # Structure: [Current State, Current Vel, Target State, Target Vel, Phase, Last Action]
        obs_dim = (self.model.nq * 2) + (self.model.nv * 2) + 1 + n_actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.viewer = None
        self.current_step_idx = 0
        
        # Track previous action for smoothness calculations
        self.last_action = np.zeros(n_actions, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset internal state
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)

        # Reference State Initialization (RSI)
        if np.random.rand() < 0.5:
            start_idx = 0
        else:
            min_left = int(0.5 / self.trajectory_dt)
            start_idx = np.random.randint(0, self.max_steps - min_left)

        self.current_step_idx = start_idx
        
        # Initialize robot state
        init_qpos = self.expert_qpos[start_idx].copy()
        init_qvel = self.expert_qvel[start_idx].copy()
        
        # Add small noise to initial state
        noise_pos = np.random.normal(0, 0.001, size=self.model.nq)
        if noise_pos[1] < 0: noise_pos[1] = 0 
        init_qpos += noise_pos
        init_qvel += np.random.normal(0, 0.01, size=self.model.nv)
        
        self.data.qpos[:] = init_qpos
        self.data.qvel[:] = init_qvel
        self.data.time = start_idx * self.trajectory_dt
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # Scale actions to control range
        low = self.ctrl_ranges[:, 0]
        high = self.ctrl_ranges[:, 1]
        scaled_action = low + (action + 1) * 0.5 * (high - low)
        self.data.ctrl[:] = scaled_action
        
        # Step simulation
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
            
        self.current_step_idx += 1

        # Detect self-collisions (contacts not involving the ground)
        self_collision_count = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if geom1 != self.ground_id and geom2 != self.ground_id:
                self_collision_count += 1
        
        # Retrieve expert target state
        current_time = self.data.time
        target_idx = int(current_time / self.trajectory_dt)
        target_idx = min(target_idx, self.max_steps - 1)

        target_qpos = self.expert_qpos[target_idx]
        target_qvel = self.expert_qvel[target_idx]
        
        # Calculate Rewards
        
        # Pose & Root Tracking
        joint_diff = self.data.qpos[2:] - target_qpos[2:]
        joint_err = np.sum(joint_diff**2)
        
        root_diff = self.data.qpos[0:2] - target_qpos[0:2]
        root_err = np.sum(root_diff**2)
        
        vel_err = np.sum((self.data.qvel - target_qvel)**2)
        
        # Reward Weights
        w_pose   = 0.60
        w_root   = 0.25
        w_vel    = 0.10
        w_energy = 0.01 
        w_smooth = 0.05
        w_coll   = 0.10  
        
        r_pose = np.exp(-2.0 * joint_err)
        r_root = np.exp(-5.0 * root_err)
        r_vel  = np.exp(-0.1 * vel_err)
        
        # Penalties (Control, Smoothness, Collision)
        control_penalty = np.sum(np.square(action)) 
        
        action_diff = action - self.last_action
        smoothness_penalty = np.sum(np.square(action_diff))
        
        collision_penalty = self_collision_count * w_coll
        
        # Combine terms
        base_reward = (w_pose * r_pose) + (w_root * r_root) + (w_vel * r_vel)
        penalty_cost = (w_energy * control_penalty) + (w_smooth * smoothness_penalty) + collision_penalty
        
        reward = base_reward - penalty_cost
        
        # Update last action
        self.last_action = action.copy()
        
        # Check termination conditions
        terminated = False
        if target_idx >= self.max_steps - 10:
            terminated = True
            
        # Early stopping for failure states
        if joint_err > 20.0: terminated = True; reward = 0
        if self.data.qpos[1] < (target_qpos[1] - 0.35): terminated = True; reward = 0
        if abs(self.data.qpos[0] - target_qpos[0]) > 1.0: terminated = True; reward = 0
        
        # Terminate on excessive self-collision
        if self_collision_count > 5:
            terminated = True
            reward = -1.0
            
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # Retrieve target state for observation
        current_time = self.data.time
        target_idx = int(current_time / self.trajectory_dt)
        target_idx = min(target_idx, self.max_steps - 1)
        
        target_qpos = self.expert_qpos[target_idx]
        target_qvel = self.expert_qvel[target_idx]
        
        phase = np.array([self.current_step_idx / self.max_steps])
        
        return np.concatenate([
            self.data.qpos, 
            self.data.qvel, 
            target_qpos, 
            target_qvel, 
            phase,
            self.last_action
        ]).astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()