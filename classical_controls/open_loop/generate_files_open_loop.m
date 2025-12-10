% -------------------------------------------------------------------------
% generate_files_fmincon.m
% -------------------------------------------------------------------------
% This is the code required to determine and setup the dynamics of a 5-link 
% robot doing a handstand. It sets up the problem symbolically,
% auto-generates necessary files.
% -------------------------------------------------------------------------
% Created by: Jason Abi Chebli
% Last Modified: 07-November-2025
% -------------------------------------------------------------------------

clear all; close all; clc;

restoredefaultpath;
if ~exist('./auto')
    mkdir('./auto');
end
addpath('./auto');

% =========================================================================
% ----------------- Calculating Dynamics (Symbollically) ------------------
% =========================================================================

%% ---------------------------- Symbolic Setup -----------------------------
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
syms th1_des th2_des th3_des th4_des th5_des

% Lengths [m]
l_L = 1; % leg
l_T = 1; % torso
l_A = 0.75; % arm

% Masses [kg]
m_L = 5; % leg
m_T = 10; % torso
m_A = 5; % arm
m_H = 10; % hip
m_S = 5; % shoulder

% Inertias [kg m^2]
J_L = 0; % leg
J_T = 0; % torso
J_A = 0; %arm

% World Parameters
g = 9.81; % gravity [m / s^2]

% Generalized coords and rates
q  = [x; y; q1; q2; q3; q4; q5];
qdot = [xdot; ydot; q1dot; q2dot; q3dot; q4dot; q5dot];

% abs angle coords
th1 = q1 + q5 - pi;
th2 = q2 + q5 - pi;
th3 = q3 + q5 - pi;
th4 = q4 + q5 - pi; 
th5 = q5;

% Output References
th_des = [th1_des; th2_des; th3_des; th4_des; th5_des];

% State vector
s = [q; qdot];

% Inputs
u = [u1; u2; u3; u4];

NDof = length(q);

%% -------------------------- Forward kinematics ---------------------------
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

matlabFunction(p_com_L1, 'File', 'auto/p_com_L1_auto', 'Vars', {s});
matlabFunction(p_com_L2, 'File', 'auto/p_com_L2_auto', 'Vars', {s});
matlabFunction(p_com_A3, 'File', 'auto/p_com_A3_auto', 'Vars', {s});
matlabFunction(p_com_A4, 'File', 'auto/p_com_A4_auto', 'Vars', {s});
matlabFunction(p_com_T, 'File', 'auto/p_com_T_auto', 'Vars', {s});

matlabFunction(p_L1, 'File', 'auto/p_L1_auto', 'Vars', {s});
matlabFunction(p_L2, 'File', 'auto/p_L2_auto', 'Vars', {s});
matlabFunction(p_A3, 'File', 'auto/p_A3_auto', 'Vars', {s});
matlabFunction(p_A4, 'File', 'auto/p_A4_auto', 'Vars', {s});
matlabFunction(p_H, 'File', 'auto/p_H_auto', 'Vars', {s});
matlabFunction(p_S, 'File', 'auto/p_S_auto', 'Vars', {s});

%% ---------------------- Lagragian Dynamics -----------------
% COM linear velocities
tic;

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
fprintf("Finished Lagrangian Dynamics in " + toc + " secs\n");

%% ----------- Stance Jacobian and Stance Jacobian Derivative --------------
% This is needed for dynamic equation with constraints
tic;

p_st = p_L1;
J_st = jacobian(p_st, q);
Jdot_st = jacobian(J_st*qdot, q);

%%%% Lecture Note Method
P = [D, -J_st'; J_st, zeros(height(J_st), height(J_st))];
Q = [-C*qdot - G + B*u; -Jdot_st*qdot];
constraint_dyn = P\Q;
F_st = constraint_dyn(NDof+1:end);

F_st_u = jacobian(F_st, u);
F_st_nu = F_st - F_st_u*u;

matlabFunction(p_st, 'File', 'auto/p_st_auto', 'Vars', {s});
matlabFunction(F_st, 'File', 'auto/F_st_auto', 'Vars', {s, u});

fprintf("Finished Stance Forces in " + toc + " secs\n");

%% ----------- Stance Impact Map ------------ %
tic;
p_sw = p_L2;
J_sw = jacobian(p_sw, q);

[postimpact] = ([D, -J_sw'; J_sw, zeros(height(J_sw), height(J_sw))])\[D*qdot; zeros(2, 1)];
qdot_st_plus = postimpact(1:NDof);

matlabFunction(p_sw, 'File', 'auto/p_sw_auto', 'Vars', {s}); 
matlabFunction(qdot_st_plus, 'File', 'auto/qdot_st_plus_auto', 'Vars', {s});

fprintf("Finished Gait Impact Forces in " + toc + " secs\n");

%% ----------------- Stance Dynamics -----------------
tic;
fdyn_st = [qdot; D\(J_st'*F_st_nu - C*qdot - G)];
gdyn_st = [zeros(height(qdot), height(u)); D\(B + J_st'*F_st_u)];

matlabFunction(fdyn_st, 'File', 'auto/fdyn_st_auto', 'Vars', {s});
matlabFunction(gdyn_st, 'File', 'auto/gdyn_st_auto', 'Vars', {s});

fprintf("Finished Stance Dynamics in " + toc + " secs\n");

%% ------------------ Stance Output Dynamics ----------------
tic;

y_st = [th5 - th5_des;
     th4 - th4_des;
     th3 - th3_des;
     th2 + th1];

% Lie Derivatives
Lfy_st = jacobian(y_st, s)*fdyn_st;
L2fy_st = jacobian(Lfy_st, s)*fdyn_st;
LgLfy_st = jacobian(Lfy_st, s)*gdyn_st;

matlabFunction(y_st, 'File', 'auto/y_st_auto', 'Vars', {s, th_des});
matlabFunction(Lfy_st, 'File', 'auto/Lfy_st_auto', 'Vars', {s, th_des});
matlabFunction(L2fy_st, 'File', 'auto/L2fy_st_auto', 'Vars', {s, th_des});
matlabFunction(LgLfy_st, 'File', 'auto/LgLfy_st_auto', 'Vars', {s, th_des});

fprintf("Finished Stance Output Dynamics in " + toc + " secs\n");

%%%%%%%%%%%%%%%%%%% Hand Plant Functions %%%%%%%%%%%%%%%%%%%%%%%%
%% ------------------- Hand Plant Impact Map -------------- %
tic;
p_hp = p_A3;
J_hp = jacobian(p_hp, q);

[postimpact] = ([D, -J_hp'; J_hp, zeros(height(J_hp), height(J_hp))])\[D*qdot; zeros(2, 1)];
qdot_hp_plus = postimpact(1:NDof);

matlabFunction(p_hp, 'File', 'auto/p_hp_auto', 'Vars', {s});
matlabFunction(qdot_hp_plus, 'File', 'auto/qdot_hp_plus_auto', 'Vars', {s});

fprintf("Finished Hand Plant Impact Forces in " + toc + " secs\n");

%% -------------------- Hand Plant Output Dynamics ---------- %
tic;
y_hp = [th5 - th5_des;
        th4 - th4_des;
        th3 - th3_des;
        (th2 - pi/2) - th1];

% Lie Derivatives
Lfy_hp = jacobian(y_hp, s)*fdyn_st;
L2fy_hp = jacobian(Lfy_hp, s)*fdyn_st;
LgLfy_hp = jacobian(Lfy_hp, s)*gdyn_st;

matlabFunction(y_hp, 'File', 'auto/y_hp_auto', 'Vars', {s, th_des});
matlabFunction(Lfy_hp, 'File', 'auto/Lfy_hp_auto', 'Vars', {s, th_des});
matlabFunction(L2fy_hp, 'File', 'auto/L2fy_hp_auto', 'Vars', {s, th_des});
matlabFunction(LgLfy_hp, 'File', 'auto/LgLfy_hp_auto', 'Vars', {s, th_des});

fprintf("Finished Hand Plant Output Dynamics in " + toc + " secs\n");

%%%%%%%%%%%%%%%%%%%%%%%%%% Swing Up %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Swing up Virtual Constraint
tic;
p_su = p_A3;
J_su = jacobian(p_su, q);
Jdot_su = jacobian(J_su*qdot, q);

%%%% Lecture Note Method
P = [D, -J_su'; J_su, zeros(height(J_su), height(J_su))];
Q = [-C*qdot - G + B*u; -Jdot_su*qdot];
constraint_dyn = P\Q;
F_su = constraint_dyn(NDof+1:end);

F_su_u = jacobian(F_su, u);
F_su_nu = F_su - F_su_u*u;

matlabFunction(F_su, 'File', 'auto/F_su_auto', 'Vars', {s, u});

fprintf("Finished Swing Up Forces in " + toc + " secs\n");

%% ----------------- Swing Up Dynamics -----------------
tic;
fdyn_su = [qdot; D\(J_su'*F_su_nu - C*qdot - G)];
gdyn_su = [zeros(height(qdot), height(u)); D\(B + J_su'*F_su_u)];

matlabFunction(fdyn_su, 'File', 'auto/fdyn_su_auto', 'Vars', {s});
matlabFunction(gdyn_su, 'File', 'auto/gdyn_su_auto', 'Vars', {s});

fprintf("Finished Swing Up Dynamics in " + toc + " secs\n");

%% ------------------- Swing Up Impact Map -------------- %
tic;
p_su_i = p_A4;
J_su_i = jacobian(p_su_i, q);

[postimpact] = ([D, -J_su_i'; J_su_i, zeros(height(J_su_i), height(J_su_i))])\[D*qdot; zeros(2, 1)];
qdot_su_i_plus = postimpact(1:NDof);

matlabFunction(p_su_i, 'File', 'auto/p_su_i_auto', 'Vars', {s});
matlabFunction(qdot_su_i_plus, 'File', 'auto/qdot_su_i_plus_auto', 'Vars', {s});

fprintf("Finished Swing Up Impact Forces in " + toc + " secs\n");

%% --------------------- Swing Up Output Dynamics ---------- %
tic;
y_su = [th5 - th5_des;
        th4 + th3;
        th2 - th2_des;
        th1 - th1_des];

% Lie Derivatives
Lfy_su = jacobian(y_su, s)*fdyn_su;
L2fy_su = jacobian(Lfy_su, s)*fdyn_su;
LgLfy_su = jacobian(Lfy_su, s)*gdyn_su;

matlabFunction(y_su, 'File', 'auto/y_su_auto', 'Vars', {s, th_des});
matlabFunction(Lfy_su, 'File', 'auto/Lfy_su_auto', 'Vars', {s, th_des});
matlabFunction(L2fy_su, 'File', 'auto/L2fy_su_auto', 'Vars', {s, th_des});
matlabFunction(LgLfy_su, 'File', 'auto/LgLfy_su_auto', 'Vars', {s, th_des});

fprintf("Finished Hand Plant Output Dynamics in " + toc + " secs\n");

