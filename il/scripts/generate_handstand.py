import numpy as np
import os
import mujoco
import mujoco.viewer
import time
from scipy.interpolate import CubicSpline

# --- CONFIGURATION ---
OUTPUT_FILE = "expert_trajectory_full.npy"
DT = 0.002           
TOTAL_TIME = 7.0     

# Keyframe Definition
# Format: [Time, RootX, RootZ, Pitch, RArm, LArm, RHip, RKnee, LHip, LKnee]
keyframes = [
    # 1. Start (0.0s): Standing Tall
    [0.0,   0.0,  0.0,    0,    160, 160,    0,   0,      0,   0],

    # 2. Squat (0.7s): Load the Springs
    # Knees bent, body drops. Feet positioned under center of mass for stability.
    [0.7,   0.0,  -0.35,    0,    150, 150,    -10,   70,    -10,   70],

    # 3. Preparation (1.1s): Shift weight and straighten back leg
    [1.1,   0.2,  -0.35,    40,    120, 100,    0,   0,    -20,   40],

    # 4. Push Off (1.5s): Power Drive
    # Strong kick with the left leg (-90 deg) to generate forward momentum.
    [1.5,   1.3,  -0.40,  90,     60,  60,    80,   0,     -90,   0],

    # 5. Handstand (3.0s): Inversion
    # Robot is fully inverted (Pitch 180). Root moves forward to balance point.
    [3.0,   2.2,  -0.35, 180,     10,  10,    90,   0,     -90,   0],

    # 6. Hold (5.0s): Maintain Balance
    [5.0,   2.2,  -0.35, 180,     10,  10,    0,   0,      0,   0],

    # 7. End (7.0s)
    [7.0,   2.2,  -0.35, 180,     10,  10,    0,   0,      0,   0],
]


# --- 1. TRAJECTORY INTERPOLATION ---
kf_data = np.array(keyframes)
times = kf_data[:, 0]
poses_deg = kf_data[:, 1:]
poses_rad = poses_deg.copy()
poses_rad[:, 2:] = np.deg2rad(poses_deg[:, 2:]) 

print("Generating Spline...")
# Use Cubic Spline to create smooth transitions between keyframes
cs = CubicSpline(times, poses_rad, bc_type='clamped')
sim_times = np.arange(0, TOTAL_TIME, DT)
qpos_traj = cs(sim_times)
qvel_traj = cs(sim_times, 1)

# --- 2. AUTO-GROUNDING ADJUSTMENT ---
def fix_ground_clipping(model, data, traj):
    """
    Adjusts the trajectory height (RootZ) frame-by-frame.
    Ensures the lowest point of the robot touches the ground with slight 
    penetration (-2mm) to ensure stable contact dynamics in MuJoCo.
    """
    print("Auto-correcting ground penetration (Target: -2mm)...")
    corrected_traj = traj.copy()
    
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) 
                for name in ["right_foot_site", "left_foot_site", "right_hand_site", "left_hand_site"]]
    head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head_geom")
    head_radius = 0.12 
    
    target_depth = -0.002 
    
    for i in range(len(corrected_traj)):
        data.qpos[:] = corrected_traj[i]
        mujoco.mj_forward(model, data) 
        
        min_z = 100.0 
        margin = 0.05 
        
        # Check sites (hands/feet)
        for sid in site_ids:
            site_z = data.site_xpos[sid][2]
            lowest_point = site_z - margin
            if lowest_point < min_z: min_z = lowest_point
        
        # Check head collision
        head_z = data.geom_xpos[head_geom_id][2]
        if (head_z - head_radius) < min_z: min_z = (head_z - head_radius)
            
        # Shift Z to meet target depth
        if min_z < target_depth:
            corrected_traj[i, 1] += (target_depth - min_z)
        elif min_z > 0.05: 
             pass 

    return corrected_traj

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "../models/recorder_model.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

qpos_traj = fix_ground_clipping(model, data, qpos_traj)

# --- SAVE TRAJECTORY ---
output_path = os.path.join(current_dir, f"../{OUTPUT_FILE}")
save_dict = {
    "qpos": qpos_traj,
    "qvel": qvel_traj,
    "time": sim_times,
    "dt": DT
}
np.save(output_path, save_dict)
print(f"Saved Physics-Corrected Trajectory to {output_path}")

# --- VISUALIZE ---
print("\nPlaying... (Mode 3 Test Ready)")
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    while viewer.is_running():
        now = time.time() - start_time
        if now > TOTAL_TIME + 0.5: 
            start_time = time.time()
            now = 0
        idx = int(now / DT)
        idx = min(idx, len(qpos_traj) - 1)
        
        data.qpos[:] = qpos_traj[idx]
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(DT)