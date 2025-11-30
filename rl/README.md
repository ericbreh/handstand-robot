# 🏋️‍♂️ 5-Link Robot Cartwheel & Handstand RL Project

This project trains a **5-link humanoid robot** (torso + 2 arms + 2 legs) to perform a **dynamic cartwheel** and transition into a **balanced handstand** using **Reinforcement Learning (PPO)**.

The simulation environment is built using **MuJoCo** and **Gymnasium**, and the agent is trained using **Stable Baselines3**.

---

## 📁 Project Structure
```
models/
└── robot_model.xml # MuJoCo physics model of the 5-link robot

envs/
└── five_link_env.py # Custom Gymnasium environment, rewards, obs space

train_ppo.py # Main PPO training script
```

---

## ⚙️ Setup & Installation

Follow these steps to set up and run the project.

---

### 1. Create a Virtual Environment

```bash
python3 -m venv handstand-robot
```

2. Activate the Virtual Environment

```bash
source handstand-robot/bin/activate
```

3. Install Dependencies

```bash
pip install gymnasium[mujoco] stable-baselines3 tensorboard numpy
```

---

## 🚀 Running the Project

### Train the PPO Agent

Runs PPO training, saves checkpoints every 100k steps, and stores the final model after 5M steps:

```bash
python train_ppo.py
```

## 📊 Track Training Progress with TensorBoard

Open a new terminal, activate the virtual environment again, then run:

```bash
tensorboard --logdir ./runs
```

Then open the displayed URL (usually 
http://localhost:6006/) in your browser to view reward curves and learning metrics.