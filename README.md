# Handstand Robot

Kinematic modelling, dynamics, and simulation of a handstand robot.

## Kinematics:
<img width="1056" height="1241" alt="image" src="https://github.com/user-attachments/assets/b24686f8-1904-445b-8cbb-5f8fb4dce155" />

## Tree: 
├───classical_controls
│   ├───animate
│   ├───auto
│   ├───dynamics
│   ├───events
│   └───media
├───documents
└───rl

## To run classical controls:
1. run generate_files.m* (this generates functions that are used in simulate_handstand.m)
3. run simulate_handstand.m

*Note if you pull the whole repo, these files should already be generated so you wont need to actually run this

## Todos:

### Classical Controls
- [ ] Try to visualize the 5-link system and seee if the kinematics was correctly setup in generate_files.m
- [ ] Determine the virtual constraints
- [ ] Determine the path of the 5-link system
- [ ] Write the dynamics for stance
- [ ] Write the dynamics for flight
- [ ] Write the dynamics for impact
- [ ] Write the event takeoff
- [ ] Write the event touchdown
- [ ] Implement basic simulation/animation

### Reinforcement Learning 
- [ ] Everything still 
