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

# best no physics version so far
# --- KEYFRAMES ---
# [Time, RootX, RootZ, Pitch,  RArm, LArm,  RHip, RKnee,  LHip, LKnee]
# keyframes = [
#     # 1. START (0.0s): Standing Tall
#     [0.0,   0.0,  0.0,    0,    160, 160,    0,   0,      0,   0],

#     # 2. SQUAT (0.8s): Load the Springs
#     # FIX: Kept Hips near 0. Previous '50' moved feet forward, causing a slip later.
#     # Knees bent (-70) -> Body drops (-0.35).
#     # Feet are now directly under the center of mass (Stable).
#     [1.0,   0.0,  -0.35,    0,    150, 150,    -10,   70,    -10,   70],

#     # 3. LUNGE ENTRY (1.6s): Step Right Foot Forward
#     # We lift the Right Leg (Hip 60) to step forward.
#     # Left Leg stays planted (Hip 0) to support weight.
#     [3.0,   0.2,  -0.30,   20,    120, 100,    60,   30,    -10,   30],

#     # 4. PUSH OFF (2.4s): The "Power Drive"
#     # Left Leg kicks HARD back (-90).
#     # Geometry Check: Leg extends 0.9m back -> Root moves 0.2 -> 1.3 (1.1m gain).
#     # This matches the physics of a push-off.
#     [4.0,   1.3,  -0.40,  90,     60,  60,    80,   0,     -90,   0],

#     # 5. HANDSTAND (4.0s): Inversion
#     # Root continues forward to 2.2m.
#     # Pitch hits 180.
#     [5.0,   2.2,  -0.35, 180,     10,  10,    110,   0,     -90,   0],

#     # 6. HOLD (6.0s)
#     [6.0,   2.2,  -0.35, 180,     10,  10,    0,   0,      0,   0],

#     # 7. END
#     [7.0,   2.2,  -0.35, 180,     10,  10,    0,   0,      0,   0],
# ]


# best physics version so far
# [Time, RootX, RootZ, Pitch,  RArm, LArm,  RHip, RKnee,  LHip, LKnee]
keyframes = [
    # 1. START (0.0s): Standing Tall
    [0.0,   0.0,  0.0,    0,    160, 160,    0,   0,      0,   0],

    # 2. SQUAT (0.8s): Load the Springs
    # FIX: Kept Hips near 0. Previous '50' moved feet forward, causing a slip later.
    # Knees bent (-70) -> Body drops (-0.35).
    # Feet are now directly under the center of mass (Stable).
    [0.7,   0.0,  -0.35,    0,    150, 150,    -10,   70,    -10,   70],

    # 3. Straighten backfoot 
    [1.1,   0.2,  -0.35,    40,    120, 100,    0,   0,    -20,   40],

    # 4. PUSH OFF (2.4s): The "Power Drive"
    # Left Leg kicks HARD back (-90).
    # Geometry Check: Leg extends 0.9m back -> Root moves 0.2 -> 1.3 (1.1m gain).
    # This matches the physics of a push-off.
    [1.5,   1.3,  -0.40,  90,     60,  60,    80,   0,     -90,   0],

    # 5. HANDSTAND (4.0s): Inversion
    # Root continues forward to 2.2m.
    # Pitch hits 180.
    [3.0,   2.2,  -0.35, 180,     10,  10,    90,   0,     -90,   0],

    # 6. HOLD (6.0s)
    [5.0,   2.2,  -0.35, 180,     10,  10,    0,   0,      0,   0],

    # 7. END
    [7.0,   2.2,  -0.35, 180,     10,  10,    0,   0,      0,   0],
]

# Best physics version
# [Time, RootX, RootZ, Pitch,  RArm, LArm,  RHip, RKnee,  LHip, LKnee]
# keyframes = [
#     # 1. START (0.0s): Standing Tall
#     [0.0,   0.0,  0.0,    0,    160, 160,    0,   0,      0,   0],

#     # 2. SQUAT (0.8s): Load the Springs
#     # FIX: Kept Hips near 0. Previous '50' moved feet forward, causing a slip later.
#     # Knees bent (-70) -> Body drops (-0.35).
#     # Feet are now directly under the center of mass (Stable).
#     [0.5,   0.0,  -0.35,    0,    150, 150,    -10,   80,    -10,   80],

#     # 3. Straighten backfoot 
#     [0.8,   0.2,  -0.35,    40,    120, 70,    0,   0,    -20,   60],

#     # 4. PUSH OFF (2.4s): The "Power Drive"
#     # Left Leg kicks HARD back (-90).
#     # Geometry Check: Leg extends 0.9m back -> Root moves 0.2 -> 1.3 (1.1m gain).
#     # This matches the physics of a push-off.
#     [1.3,   0.8,  -0.40,  90,    80,  60,    80,   0,     -90,   0],

#     # 5. HANDSTAND (4.0s): Inversion
#     # Root continues forward to 2.2m.
#     # Pitch hits 180.
#     [2.6,   1.5,  -0.35, 180,     20,  20,    80,   0,     -80,   0],

#     # 6. HOLD (6.0s)
#     [5.0,   1.5,  -0.35, 180,     10,  10,    0,   0,      0,   0],

#     # 7. END
#     [7.0,   1.5,  -0.35, 180,     10,  10,    0,   0,      0,   0],
# ]

# --- 1. SPLINE GENERATION ---
kf_data = np.array(keyframes)
times = kf_data[:, 0]
poses_deg = kf_data[:, 1:]
poses_rad = poses_deg.copy()
poses_rad[:, 2:] = np.deg2rad(poses_deg[:, 2:]) 

print("Generating Spline...")
cs = CubicSpline(times, poses_rad, bc_type='clamped')
sim_times = np.arange(0, TOTAL_TIME, DT)
qpos_traj = cs(sim_times)
qvel_traj = cs(sim_times, 1)

# --- 2. AUTO-GROUNDING (REALISTIC) ---
def fix_ground_clipping(model, data, traj):
    print("Auto-correcting ground penetration (Target: -2mm)...")
    corrected_traj = traj.copy()
    
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) 
                for name in ["right_foot_site", "left_foot_site", "right_hand_site", "left_hand_site"]]
    head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head_geom")
    head_radius = 0.12 
    
    # REALISM FIX: 
    # -0.002 (2 millimeters) is enough to activate friction without exploding.
    # It ensures the 'soft contact' margin is triggered.
    target_depth = -0.002 
    
    for i in range(len(corrected_traj)):
        data.qpos[:] = corrected_traj[i]
        mujoco.mj_forward(model, data) 
        
        min_z = 100.0 
        margin = 0.05 
        
        for sid in site_ids:
            site_z = data.site_xpos[sid][2]
            lowest_point = site_z - margin
            if lowest_point < min_z: min_z = lowest_point
        
        head_z = data.geom_xpos[head_geom_id][2]
        if (head_z - head_radius) < min_z: min_z = (head_z - head_radius)
            
        # Shift so the lowest point is exactly at target_depth
        if min_z < target_depth:
            corrected_traj[i, 1] += (target_depth - min_z)
        # OPTIONAL: If we are floating high above, pull us down too!
        # This keeps feet glued even if they lift slightly in the spline.
        elif min_z > 0.05: 
             # (Only correct if we aren't intentionally jumping)
             pass 

    return corrected_traj

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "../models/recorder_model.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

qpos_traj = fix_ground_clipping(model, data, qpos_traj)

# --- SAVE ---
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