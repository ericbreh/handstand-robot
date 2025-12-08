import mujoco
import mujoco.viewer
import time
import os

# Define the path to your MuJoCo XML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models', 'recorder_model.xml')

def visualize_model():
    """
    Loads the MuJoCo model using the 'mujoco' library to verify the XML is valid and visualize the robot.
    """
    print(f"Attempting to load model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("ERROR: XML file not found at the specified path.")
        return

    try:
        # Load the model and data
        # This is the most direct way to check if the XML is valid
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        
        print("Model loaded successfully! XML is valid.")
        print("Launching MuJoCo viewer... (Press ESC to exit)")

        # Launch the viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            
            # Simple simulation loop
            while viewer.is_running():
                # Step the physics
                mujoco.mj_step(model, data)
                
                # Sync the viewer with the physics state
                viewer.sync()
                
                # Slow down to make it viewable (approx 60 FPS)
                time.sleep(1.0/60.0)

    except Exception as e:
        print(f"\nERROR: Could not load the MuJoCo model.")
        print(f"Details: {e}")

if __name__ == "__main__":
    visualize_model()