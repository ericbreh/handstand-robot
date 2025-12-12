# 🏋️‍♂️ 7-Link Robot Cartwheel & Handstand Imitation Learning (IL) Project

This project trains a **7-link humanoid robot** (torso + 2 arms + 2 legs **with knees**) to perform a **dynamic cartwheel** and transition into a **balanced handstand** using **RL (PPO) with Reference Motion (**.

The simulation environment is built using **MuJoCo** and **Gymnasium**, and the agent is trained using **Stable Baselines3**.

---

## 📁 Project Structure
```
.
├── models/
│   └── recorder_model.xml    # MuJoCo physics model of the 7-link robot
├── envs/
│   └── handstand_env.py      # Custom Gymnasium environment, rewards, obs space
├── scripts/
│   ├── generate_handstand.py # Generates and visualizes the reference trajectory
│   ├── play.py               # Visualizer: see the reference, random policy, or trained agent
│   ├── train.py              # Main PPO training script
│   └── visualize_model.py    # Sandbox to test model joints and degrees of freedom
└── expert_trajectory_full.npy # The saved reference trajectory (teacher policy)
```

---

## ⚙️ Setup & Installation

Follow these steps to set up and run the project.

---

### 1. Create a Virtual Environment

```bash
python3 -m venv handstand-robot
```

### 2. Activate the Virtual Environment

```bash
source handstand-robot/bin/activate
```

### 3. Install Dependencies
We use `scipy` for trajectory interpolation and `stable-baselines3` for the PPO implementation.

```bash
pip install gymnasium[mujoco] scipy stable-baselines3 tensorboard numpy
```

---

## 🚀 Running the Project

### Train the PPO Agent

Runs PPO training, saves checkpoints every 100k steps, and stores the final model after 2M steps:

```bash
python scripts/train.py
```

### Visualize the Result

To see your trained robot in action (or to debug the reference motion):

```bash
python scripts/play.py
```


## 📊 Track Training Progress with TensorBoard

Open a new terminal, activate the virtual environment again, then run:

```bash
tensorboard --logdir ./runs
```

Then open the displayed URL (usually 
http://localhost:6006/) in your browser to view reward curves and learning metrics.