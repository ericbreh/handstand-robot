import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import select

# --- CONFIGURATION ---
GRAVITY_Z = -9.81  # REAL Earth gravity. Helps you see if the pose is stable.

# --- 1. Load the XML and Patch it for Posing ---
current_dir = os.path.dirname(os.path.abspath(__file__))
original_model_path = os.path.join(current_dir, "../models/recorder_model.xml")

if not os.path.exists(original_model_path):
    raise FileNotFoundError(f"Missing: {original_model_path}")

print("Loading XML and converting 'Free Joint' to 'Sliders' for posing...")

with open(original_model_path, 'r') as f:
    xml_content = f.read()

# HACK: The training model uses a <freejoint> (7 DOF), but for recording 
# we need separate sliders (X, Z, Pitch) to control it easily.
# We replace the freejoint with 3 distinct joints just for this session.
if '<freejoint name="root"/>' in xml_content:
    print(" -> Patching <freejoint> -> <slide/hinge> for control...")
    # Replace freejoint with 2 slides (X, Z) and 1 hinge (Pitch)
    posing_joints = """
    <slide name="root_x"     axis="1 0 0" pos="0 0 0"/>
    <slide name="root_z"     axis="0 0 1" pos="0 0 0"/>
    <hinge name="root_pitch" axis="0 1 0" pos="0 0 0"/>
    """
    modified_xml = xml_content.replace('<freejoint name="root"/>', posing_joints)
else:
    print(" -> Warning: No <freejoint> found. Assuming model already has slides.")
    modified_xml = xml_content

# Now inject the "Fake Motors" to control these new joints
fake_motors = """
    <position name="pose_root_x"     joint="root_x"     kp="1000" ctrlrange="-5 5"/>
    <position name="pose_root_z"     joint="root_z"     kp="1000" ctrlrange="-2 3"/>
    <position name="pose_root_pitch" joint="root_pitch" kp="1000" ctrlrange="-6.28 6.28"/>
"""

# Insert motors into the <actuator> block
if "</actuator>" in modified_xml:
    modified_xml = modified_xml.replace("</actuator>", fake_motors + "\n    </actuator>")
else:
    # Create actuator block if missing
    modified_xml = modified_xml.replace("</mujoco>", "<actuator>" + fake_motors + "</actuator>\n</mujoco>")

# --- 2. Load the Posing Model ---
model = mujoco.MjModel.from_xml_string(modified_xml)
data = mujoco.MjData(model)

# Set Real Gravity so we can see balance issues
model.opt.gravity[:] = [0, 0, GRAVITY_Z]

# --- 3. Helper: Convert Slider Data (3-DOF) back to FreeJoint Data (7-DOF) ---
def convert_pose_to_freejoint_format(qpos_posing):
    """
    Posing Model: [root_x, root_z, root_pitch, ...joints...]
    Target Model: [root_x, root_y, root_z, qw, qx, qy, qz, ...joints...]
    """
    # Extract Root parts
    rx = qpos_posing[0]
    rz = qpos_posing[1]
    pitch = qpos_posing[2]
    
    # Rest of the body (arms/legs)
    body_joints = qpos_posing[3:]
    
    # Convert Pitch Angle to Quaternion (w, x, y, z)
    # Rotating around Y-axis (0, 1, 0)
    half_angle = pitch / 2.0
    qw = np.cos(half_angle)
    qx = 0.0
    qy = np.sin(half_angle)
    qz = 0.0
    
    # Construct the 7-DOF root
    # Note: Training env usually expects freejoint at indices 0-6
    root_7dof = np.array([rx, 0.0, rz, qw, qx, qy, qz])
    
    return np.concatenate([root_7dof, body_joints])

# --- 4. Interactive Loop ---
keyframes = []

print("\n========================================")
print("      SMART POSING TOOL (SERVO MODE)    ")
print("========================================")
print("1. Open 'Control' tab in viewer.")
print("2. 'pose_root_z' = HEIGHT. 'pose_root_pitch' = ROTATION.")
print("3. CHECK FEET: If they float, lower Z. If buried, raise Z.")
print("4. CHECK BALANCE: If robot tips over, your pose is unbalanced!")
print("----------------------------------------")
print("5. Press [ENTER] in this terminal to capture frame.")
print("6. Type 'q' + [ENTER] to save and quit.")
print("========================================\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Step physics (Servos hold the pose)
        mujoco.mj_step(model, data)
        viewer.sync()

        # Handle Input
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if 'q' in line:
                break
            
            # CAPTURE
            print(f"Captured Keyframe #{len(keyframes) + 1}")
            
            # Convert data for the training environment
            # We zero out velocities for keyframes
            raw_qpos = data.qpos.copy()
            clean_qpos = convert_pose_to_freejoint_format(raw_qpos)
            clean_qvel = np.zeros(model.nv + (6 - 3)) # Adjust velocity dim (approx)
            # Actually, simpler to just save 0s of correct size later or let interpolate handle it
            # But let's save the clean qpos specifically.
            
            keyframes.append((clean_qpos, np.zeros_like(clean_qpos))) # Placeholder vel
            print(" -> Pose Saved (and converted to FreeJoint format)!")

# --- 5. Save Data ---
if len(keyframes) > 0:
    output_path = os.path.join(current_dir, "../expert_data.npy")
    np.save(output_path, np.array(keyframes, dtype=object))
    print(f"\nSaved {len(keyframes)} poses to {output_path}")
    print("Ready for interpolation!")