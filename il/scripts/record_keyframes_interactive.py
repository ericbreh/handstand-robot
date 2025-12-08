import mujoco
import mujoco.viewer
import numpy as np
import os
import time

# --- CONFIGURATION ---
OUTPUT_PATH = "expert_data.npy"
FPS = 30.0                 
FRAME_TIME = 1.0 / FPS

# --- 1. Load Model ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check both potential locations
possible_paths = [
    os.path.join(current_dir, "../models/recorder_model.xml"),
    os.path.join(current_dir, "models/recorder_model.xml")
]

xml_path = None
for p in possible_paths:
    if os.path.exists(p):
        xml_path = p
        break

if xml_path is None:
    raise FileNotFoundError(f"Could not find recorder_model.xml in {possible_paths}")

print(f"Loading model: {xml_path}")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# --- State Variables ---
keyframes = []
gravity_on = True

# We use a dictionary to share state between the callback and the main loop
# This avoids the "IndexError" you saw earlier.
app_state = {
    "flash_trigger": False
}

# Helper: Find floor material for Flash Effect
floor_mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatPlane")
if floor_mat_id >= 0:
    original_floor_rgba = model.mat_rgba[floor_mat_id].copy()
else:
    original_floor_rgba = None

def flash_floor(viewer):
    """Blinks the floor green to confirm capture."""
    if floor_mat_id == -1: return

    # Flash Green
    model.mat_rgba[floor_mat_id] = [0.2, 1.0, 0.2, 1.0]
    viewer.sync()
    time.sleep(0.1) 
    
    # Restore
    model.mat_rgba[floor_mat_id] = original_floor_rgba
    viewer.sync()

def key_callback(keycode):
    """
    Handles key presses from the viewer window.
    """
    global gravity_on
    
    # [R] (82) -> Toggle Gravity
    if keycode == 82:
        gravity_on = not gravity_on
        if gravity_on:
            model.opt.gravity[:] = [0, 0, -9.81]
            print(">> Gravity: ON")
        else:
            model.opt.gravity[:] = [0, 0, 0]
            data.qvel[:] = 0 # Stop movement
            print(">> Gravity: OFF (Float Mode)")

    # [SPace] (32) -> Capture Frame
    elif keycode == 32: 
        # 1. Capture Data
        kf_pos = data.qpos.copy()
        kf_vel = np.zeros_like(data.qvel)
        keyframes.append((kf_pos, kf_vel))
        
        print(f">> CAPTURED Frame #{len(keyframes)}")
        
        # 2. Set Python Flag for Flash
        app_state["flash_trigger"] = True

    # [S] (83) -> Save
    elif keycode == 83:
        if len(keyframes) > 0:
            full_path = os.path.join(current_dir, OUTPUT_PATH)
            np.save(full_path, np.array(keyframes, dtype=object))
            print(f"\n>> SAVED {len(keyframes)} frames to {full_path}")
            print(">> You can now run 'interpolate_trajectory.py'")
        else:
            print(">> No frames to save!")

    # [X] (88) -> Reset
    elif keycode == 88:
        mujoco.mj_resetData(model, data)
        print(">> Reset Simulation")

print("\n==========================================")
print("   INTERACTIVE RECORDER (CRASH-PROOF)     ")
print("==========================================")
print("  [ Space ]   : CAPTURE Frame (Green Flash)")
print("  [ R ] : Toggle Gravity")
print("  [ S ]   : SAVE to file")
print("  [ X ]   : RESET")
print("==========================================\n")

# --- Launch Viewer ---
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    
    # Start with Gravity ON
    model.opt.gravity[:] = [0, 0, -9.81]
    
    while viewer.is_running():
        # Check Python Flag
        if app_state["flash_trigger"]:
            flash_floor(viewer)
            app_state["flash_trigger"] = False # Reset flag

        # Step Physics
        mujoco.mj_step(model, data)
        
        # Dampen velocity if gravity is off (stops drifting)
        if not gravity_on:
            data.qvel[:] *= 0.9
            
        viewer.sync()
        time.sleep(FRAME_TIME)