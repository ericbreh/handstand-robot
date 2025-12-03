% -------------------------------------------------------------------------
% generate_files.m
% -------------------------------------------------------------------------
% This is the code required to determine and setup the dynamics of a 5-link 
% robot doing a handstand. It sets up the problem symbolically,
% auto-generates necessary files.
% -------------------------------------------------------------------------
% Created by: Jason Abi Chebli
% Last Modified: 07-November-2025
% -------------------------------------------------------------------------

clear all; close all; clc;

% =========================================================================
% ----------------- Calculating Dynamics (Symbollically) ------------------
% =========================================================================

% ---------------------------- Symbolic Setup -----------------------------
% Define Symbolic Variables for 5-link system
% x: x position of hip joint
% y: y position of hip joint
% q1: stance leg angle
% q2: swing leg angle
% q3: right arm (initially) angle
% q4: left arm (initially) angle
% q5: torso angle
syms x y q1 q2 q3 q4 q5 real
syms xdot ydot q1dot q2dot q3dot q4dot q5dot real
syms u1 u2 u3 u4 real

% Lengths [m]
l_L = 1; % leg
l_T = 0.5; % torso
l_A = 0.75; % arm

% Masses [kg]
m_L = 2; % leg
m_T = 4; % torso
m_A = 1; % arm
m_H = 5; % hip
m_S = 1; % shoulder

% Inertias [kg m^2]
J_L = 0; % leg
J_T = 0; % torso
J_A = 0; %arm

% World Parameters
g = 9.81; % gravity [m / s^2]

% Generalized coords and rates
q  = [x; y; q1; q2; q3; q4; q5];
qdot = [xdot; ydot; q1dot; q2dot; q3dot; q4dot; q5dot];

% State vector
s = [q; qdot];

% Inputs
u = [u1; u2; u3; u4];

NDoF = length(q);

% -------------------------- Forward kinematics ---------------------------
% COM Positions
p_H = [x; 
       y];

p_S = [x + l_T*sin(q5);
       y + l_T*cos(q5)];

p_com_T = [x + l_T/2*sin(q5);
               y + l_T/2*cos(q5)];

p_com_L1 = [x + l_L/2*sin(q1 + q5);
            y + l_L/2*cos(q1 + q5)];

p_com_L2 = [x + l_L/2*sin(q2 + q5);
            y + l_L/2*cos(q2 + q5)];

p_com_A3 = [x + l_T*sin(q5) + l_A/2*sin(q3 + q5);
            y + l_T*cos(q5) + l_A/2*cos(q3 + q5)];

p_com_A4 = [x + l_T*sin(q5) + l_A/2*sin(q4 + q5);
            y + l_T*cos(q5) + l_A/2*cos(q4 + q5)];

% End-effector Positions
p_L1 = [x + l_L*sin(q1 + q5);
        y + l_L*cos(q1 + q5)];

p_L2 = [x + l_L*sin(q2 + q5);
        y + l_L*cos(q2 + q5)];


p_A3 = [x + l_T*sin(q5) + l_A*sin(q3 + q5);
        y + l_T*cos(q5) + l_A*cos(q3 + q5)];

p_A4 = [x + l_T*sin(q5) + l_A*sin(q4 + q5);
        y + l_T*cos(q5) + l_A*cos(q4 + q5)];



% ----------------------------- Velocities --------------------------------
% COM linear velocities
pdot_H = simplify(jacobian(p_H, q)*qdot);
pdot_S = simplify(jacobian(p_S, q)*qdot);

pdot_com_T = simplify(jacobian(p_com_T, q)*qdot);

pdot_com_L1 = simplify(jacobian(p_com_L1, q)*qdot);
pdot_com_L2 = simplify(jacobian(p_com_L2, q)*qdot);

pdot_com_A3 = simplify(jacobian(p_com_A3, q)*qdot);
pdot_com_A4 = simplify(jacobian(p_com_A4, q)*qdot);

% Absolute angular velocities
q1dot_abs = q1dot + q5dot;
q2dot_abs = q2dot + q5dot;
q3dot_abs = q3dot + q5dot;
q4dot_abs = q4dot + q5dot;
q5dot_abs = q5dot;

% --------------------- Kinetic and Potential Energy ----------------------
% Note for potential and kinetic energy we need to use position and
% velocity of COM of each link.

KE_H = 0.5*m_H*norm(pdot_H)^2;
KE_S = 0.5*m_S*norm(pdot_S)^2;

KE_T = 0.5*m_T*norm(pdot_com_T)^2;

KE_L1 = 0.5*m_L*norm(pdot_com_L1)^2;
KE_L2 = 0.5*m_L*norm(pdot_com_L2)^2;

KE_A3 = 0.5*m_A*norm(pdot_com_A3)^2;
KE_A4 = 0.5*m_A*norm(pdot_com_A4)^2;

% Total kinetic energy
KE = simplify(KE_H + KE_S + KE_T + KE_L1 + KE_L2 + KE_A3 + KE_A4);

% potential energy

PE_H = m_H*g*p_H(2);
PE_S = m_S*g*p_S(2);

PE_T = m_T*g*p_com_T(2);

PE_L1 = m_L*g*p_com_L1(2);
PE_L2 = m_L*g*p_com_L2(2);

PE_A3 = m_A*g*p_com_A3(2);
PE_A4 = m_A*g*p_com_A4(2);

% Total potential energy
PE = simplify(PE_H + PE_S + PE_T + PE_L1 + PE_L2 + PE_A3 + PE_A4);

% Lagrangian
L = KE - PE;

% ------------------------ Lagrangian Dynamics ----------------------------
q_act = [q1; q2; q3; q4];
[D, C, G, B] = LagrangianDynamics(KE, PE, q, qdot, q_act);


% ----------- Stance Jacobian and Stance Jacobian Derivative --------------
% This is needed for dynamic equation with constraints
p_st = p_L1;

J_st = jacobian(p_st, q);
Jdot_st = simplify(jacobian(J_st*qdot, q));

P = [D, -J_st'; J_st, zeros(height(J_st), height(J_st))];
Q = [-C*qdot - G + B*u; -Jdot_st*qdot];
constraint_dyn = P\Q;

F_st = constraint_dyn(NDof+1:end);

F_st_u = simplify(jacobian(F_st, u));
F_st_nu = simplify(F_st - F_st_u*u);

% ----------- Impact Map ------------ %
p_sw = p_L2;

J_sw = jacobian(p_sw, q);

[postimpact] = ([D, -J_sw'; J_sw, zeros(height(J_sw), height(J_sw))])\[D*dq; zeros(2, 1)];
qdot_plus = simplify(postimpact(1:NDof));


% Stance Dynamics
fdyn = simplify([qdot; D\(J_st'*F_st_nu - C*qdot - G)]);
gdyn = simplify([zeros(height(qdot), height(u)); D\(B + J_st'*F_st_u)]);

% Output Dynamics

%% Export functions
if ~exist('./auto')
    mkdir('./auto');
end
addpath('./auto');
addpath('./animate');
addpath('./dynamics');
addpath('./events');

matlabFunction(F_st, 'File', 'auto/F_st_auto', 'Vars', {s});
matlabFunction(qdot_plus, 'File', 'auto/qdot_plus_auto', 'Vars', {s});

matlabFunction(p_st, 'File', 'auto/p_st_auto', 'Vars', {s});
matlabFunction(p_sw, 'File', 'auto/p_sw_auto', 'Vars', {s});

matlabFunction(p_com_L1, 'File', 'auto/p_com_L1_auto', 'Vars', {s});
matlabFunction(p_com_L2, 'File', 'auto/p_com_L2_auto', 'Vars', {s});
matlabFunction(p_com_A3, 'File', 'auto/p_com_A3_auto', 'Vars', {s});
matlabFunction(p_com_A4, 'File', 'auto/p_com_A4_auto', 'Vars', {s});
matlabFunction(p_com_T, 'File', 'auto/p_com_T_auto', 'Vars', {s});

matlabFunction(p_L1, 'File', 'auto/p_L1_auto', 'Vars', {s});
matlabFunction(p_L2, 'File', 'auto/p_L2_auto', 'Vars', {s});
matlabFunction(p_A3, 'File', 'auto/p_A3_auto', 'Vars', {s});
matlabFunction(p_A4, 'File', 'auto/p_A4_auto', 'Vars', {s});

matlabFunction(fdyn, 'File', 'auto/fdyn_auto', 'Vars', {s});
matlabFunction(gdyn, 'File', 'auto/gdyn_auto', 'Vars', {s});

