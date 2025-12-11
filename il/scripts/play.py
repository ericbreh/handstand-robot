import time
import sys
import os
import numpy as np
import mujoco
import mujoco.viewer

# Configuration
XML_PATH = "models/recorder_model.xml"
TRAJ_PATH = "expert_trajectory_full.npy"
# Update this to your latest checkpoint
MODEL_PATH = "runs/checkpoints/handstand_model_10800000_steps.zip" 

sys.path.append(os.getcwd())

# Attempt imports for Policy Mode
try:
    from envs.handstand_env import HandstandEnv
    HAS_ENV = True
except ImportError as e:
    HAS_ENV = False
    print(f"Warning: Could not import HandstandEnv ({e}). Policy mode disabled.")

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: Stable Baselines3 not found. Policy mode disabled.")

# Helper: Load Trajectory Data
def load_traj_data():
    try:
        raw_data = np.load(TRAJ_PATH, allow_pickle=True)
        if raw_data.ndim == 0: data = raw_data.item()
        else: data = raw_data
        
        play_dt = data.get('dt', 0.002) 
        if 'qpos' in data: states = data['qpos']
        else: states = list(data.values())[0]
        
        return states, play_dt
    except Exception as e:
        print(f"Error loading {TRAJ_PATH}: {e}")
        return None, None

# Mode 1: Visual Replay (Ghost Mode)
# Teleports the robot through the trajectory frames without physics integration.
def replay_trajectory_visual():
    states, play_dt = load_traj_data()
    if states is None: return

    if not os.path.exists(XML_PATH):
        print(f"Error: Could not find model at {XML_PATH}")
        return

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    print(f"Replaying {len(states)} frames (Visual Only - Looping)...")
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            # Play one full trajectory loop
            for i, state in enumerate(states):
                loop_start = time.time()
                if not viewer.is_running(): break
                
                # Teleport state
                n_qpos = m.nq
                d.qpos[:] = state[:n_qpos]
                
                mujoco.mj_forward(m, d)
                viewer.sync()
                
                # Sync playback speed
                process_time = time.time() - loop_start
                sleep_time = play_dt - process_time
                if sleep_time > 0: time.sleep(sleep_time)
            
            # Pause briefly before restarting loop
            time.sleep(0.5)

# Mode 2: Run Trained Policy
# Executes the PPO agent within the Gymnasium environment with full physics.
def run_policy():
    if not HAS_ENV or not HAS_SB3: return
    if not os.path.exists(MODEL_PATH):
        print(f"Missing model: {MODEL_PATH}"); return

    env = HandstandEnv(render_mode="human")
    model = PPO.load(MODEL_PATH)
    obs, _ = env.reset()
    
    print("Running AI Policy... (Physics ON)")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(action)
            env.render()
            if done or trunc: obs, _ = env.reset()
            time.sleep(0.002)
    finally:
        env.close()

# Mode 3: Physics Validation
# Attempts to drive motors to match the recording using direct control inputs.
def test_physics_feasibility():
    states, play_dt = load_traj_data()
    if states is None: return

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    print("\n-------------------------------------------")
    print(" PHYSICS VALIDATION TEST (LOOPING)")
    print(" We will drive the motors to match the recording.")
    print(" Gravity is ON. The Root is FREE.")
    print("-------------------------------------------\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            # Reset physics to start frame
            mujoco.mj_resetData(m, d)
            d.qpos[:] = states[0]
            d.qvel[:] = 0
            mujoco.mj_forward(m, d) 
            
            # Run loop
            for i, target_state in enumerate(states):
                loop_start = time.time()
                if not viewer.is_running(): break

                # Extract Target Joints (Indices 3 onwards)
                target_joints = target_state[3:]
                
                if len(target_joints) == m.nu:
                    d.ctrl[:] = target_joints
                
                # Step Physics
                mujoco.mj_step(m, d)
                viewer.sync()
                
                # Sync playback speed
                process_time = time.time() - loop_start
                sleep_time = play_dt - process_time
                if sleep_time > 0: time.sleep(sleep_time)
            
            # Pause before resetting for the next loop
            time.sleep(1.0)

# Main Execution
if __name__ == "__main__":
    print("\nSelect Mode:")
    print("1. Replay Expert Trajectory (Visual Ghost)")
    print("2. Run Trained PPO Policy (AI Brain)")
    print("3. Physics Feasibility Test (Motor Tracking)")
    
    choice = input("Enter 1, 2, or 3: ")
    
    if choice == "1":
        replay_trajectory_visual()
    elif choice == "2":
        run_policy()
    elif choice == "3":
        test_physics_feasibility()
    else:
        print("Invalid selection.")