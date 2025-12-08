import numpy as np
import os
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# --- Settings ---
TOTAL_TIME  = 3.5     # Time for the recorded movement
BRAKE_TIME  = 0.10     # Time to smoothly slow down to 0
HOLD_TIME   = 2.0     # Time to hold the static pose
DT          = 0.001   # Simulation timestep

# --- 1. Load Data ---
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, "../expert_data.npy")
output_path = os.path.join(current_dir, "../expert_trajectory_full.npy")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Missing {input_path}")

# Load keyframes
data = np.load(input_path, allow_pickle=True)
key_qpos_raw = np.array([frame[0] for frame in data])
n_raw_frames, n_joints = key_qpos_raw.shape
print(f"Loaded {n_raw_frames} raw keyframes.")

# --- 2. Add "Braking" Keyframe ---
# Keytimes for the raw dots
key_times_raw = np.linspace(0, TOTAL_TIME, n_raw_frames)

# We create a new timeline that includes the braking point
brake_time = TOTAL_TIME + BRAKE_TIME
last_pose = key_qpos_raw[-1]

# These are the "Dots" we will plot later
key_times_all = np.concatenate([key_times_raw, [brake_time]])
key_qpos_all  = np.vstack([key_qpos_raw, last_pose])

# --- 3. Cubic Spline Interpolation ---
# We use 'clamped' to force velocity to be zero at start and end
cs = CubicSpline(key_times_all, key_qpos_all, axis=0, bc_type='clamped')

move_duration = TOTAL_TIME + BRAKE_TIME
move_times = np.arange(0, move_duration, DT)

move_qpos = cs(move_times)
move_qvel = cs(move_times, 1)

# --- 4. Add Hold Phase ---
n_hold_frames = int(HOLD_TIME / DT)
hold_qpos = np.tile(last_pose, (n_hold_frames, 1))
hold_qvel = np.zeros((n_hold_frames, n_joints))

# Combine
final_qpos = np.vstack([move_qpos, hold_qpos])
final_qvel = np.vstack([move_qvel, hold_qvel])

# Full Timeline
total_duration = move_duration + HOLD_TIME
final_times = np.arange(0, total_duration, DT)

# Trim to match sizes
min_len = min(len(final_times), len(final_qpos))
final_times = final_times[:min_len]
final_qpos = final_qpos[:min_len]
final_qvel = final_qvel[:min_len]

# --- 5. Visualize (Restored) ---
plt.figure(figsize=(12, 10))

# Plot 1: Height (Z)
plt.subplot(3, 1, 1)
plt.plot(final_times, final_qpos[:, 1], label="Smoothed Trajectory")
# PLOT THE CHECKPOINTS (Red Dots)
plt.plot(key_times_all, key_qpos_all[:, 1], 'ro', label="Keyframes (Checkpoints)")
plt.axvline(x=TOTAL_TIME, color='orange', linestyle='--', label="Raw Data End")
plt.axvline(x=TOTAL_TIME + BRAKE_TIME, color='green', linestyle='--', label="Hold Start")
plt.title("Height (Z) - Checkpoints are Red Dots")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Pitch (Angle) - RESTORED
plt.subplot(3, 1, 2)
plt.plot(final_times, final_qpos[:, 2], label="Smoothed Angle")
plt.plot(key_times_all, key_qpos_all[:, 2], 'ro') # Dots here too
plt.axvline(x=TOTAL_TIME, color='orange', linestyle='--')
plt.axvline(x=TOTAL_TIME + BRAKE_TIME, color='green', linestyle='--')
plt.title("Pitch (Angle)")
plt.grid(True, alpha=0.3)

# Plot 3: Velocity
plt.subplot(3, 1, 3)
plt.plot(final_times, final_qvel[:, 1], label="Z Velocity")
plt.axvline(x=TOTAL_TIME, color='orange', linestyle='--')
plt.axvline(x=TOTAL_TIME + BRAKE_TIME, color='green', linestyle='--')
plt.title("Velocity (Look for smooth ramp down between Orange and Green lines)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 6. Save ---
save_dict = {
    "qpos": final_qpos,
    "qvel": final_qvel,
    "time": final_times,
    "dt": DT
}

np.save(output_path, save_dict)
print(f"Saved smoothed trajectory to: {output_path}")