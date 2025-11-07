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
syms x y phi q1 q2 q3 q4 real
syms xdot ydot phidot q1dot q2dot q3dot q4dot real
syms xddot yddot phiddot q1ddot q2ddot q3ddot q4ddot real
syms mt It wt lt dtx dty real
syms m1 I1 l1 d1 real
syms m2 I2 l2 d2 real
syms m3 I3 l3 d3 real
syms m4 I4 l4 d4 real
syms g real
Pi = sym(pi);

% Generalized coords and rates
q  = [x; y; phi; q1; q2; q3; q4];
dq = [xdot; ydot; phidot; q1dot; q2dot; q3dot; q4dot];
ddq = [xddot; yddot; phiddot; q1ddot; q2ddot; q3ddot; q4ddot];

% ----------------------- Torso Rotation Matrix ---------------------------
% Define rotation matrix of torso
R = [cos(phi)  -sin(phi);
     sin(phi)   cos(phi)];

% Define time derivative rotation matrix of torso
Rdot = phidot*[-sin(phi)    -cos(phi);
                cos(phi)    -sin(phi)]; 

% -------------------------- Forward kinematics ---------------------------
% COM of torso
pt = [x;y] + R*[dtx; dty];

% COM of link1 (right arm)
p1 = [x;y] + R*[ wt/2 + d1*sin(-q1);
                 lt/2 + d1*cos(-q1)];

% COM of link2 (left arm)
p2 = [x;y] + R*[ -wt/2 - d2*sin(q2);
                 lt/2 + d2*cos(q2)];

% COM of link3 (right leg)
p3 = [x;y] + R*[ wt/2 + d3*sin(q3);
                 -lt/2 - d3*cos(q3)];

% COM of link4 (left leg)
p4 = [x;y] + R*[ -wt/2 - d4*sin(-q4);
                 -lt/2 - d4*cos(-q4)];

% Right hand position
p_rhand = [x;y] + R*[ wt/2 + l1*sin(-q1);
                 lt/2 + l1*cos(-q1)];

% Left hand position
p_lhand = [x;y] + R*[ -wt/2 - l2*sin(q2);
                 lt/2 + l2*cos(q2)];

% Right foot position
p_rfoot = [x;y] + R*[ wt/2 + l3*sin(q3);
                 -lt/2 - l3*cos(q3)];

% Left foot position
p_lfoot = [x;y] + R*[ -wt/2 - l4*sin(-q4);
                 -lt/2 - l4*cos(-q4)];

% Total System COM
pCOM = [ (mt * pt(1) + m1 * p1(1) + m2 * p2(1) + m3 * p3(1) + m4 * p4(1))/(mt + m1 + m2 + m3 + m4);
         (mt * pt(2) + m1 * p1(2) + m2 * p2(2) + m3 * p3(2) + m4 * p4(2))/(mt + m1 + m2 + m3 + m4)];


% ----------------------------- Velocities --------------------------------
% linear velocities of COMs calcualted using Jacobian: v = J_q(p) * dq
Jpt = jacobian(pt, q);    % 2x7
Jp1 = jacobian(p1, q);
Jp2 = jacobian(p2, q);
Jp3 = jacobian(p3, q);
Jp4 = jacobian(p4, q);

vt  = simplify(Jpt * dq); % 2x1
v1  = simplify(Jp1 * dq);
v2  = simplify(Jp2 * dq);
v3  = simplify(Jp3 * dq);
v4  = simplify(Jp4 * dq);

% angular velocities definition
omegat = phidot;
omega1 = phidot - q1dot;
omega2 = phidot + q2dot;
omega3 = phidot + q3dot;
omega4 = phidot - q4dot;

% --------------------- Kinetic and Potential Energy ----------------------
% Note for potential and kinetic energy we need to use position and
% velocity of COM of each link.

% kinetic energy
Tt = simplify( (1/2)*mt*(vt.'*vt) + (1/2)*It*omegat^2 );
T1 = simplify( (1/2)*m1*(v1.'*v1) + (1/2)*I1*omega1^2 );
T2 = simplify( (1/2)*m2*(v2.'*v2) + (1/2)*I2*omega2^2 );
T3 = simplify( (1/2)*m3*(v3.'*v3) + (1/2)*I3*omega3^2 );
T4 = simplify( (1/2)*m4*(v4.'*v4) + (1/2)*I4*omega4^2 );
T  = simplify( Tt + T1 + T2 + T3 + T4 );

% potential energy
Ut = mt * g * pt(2); % p(2) is the y-component (height)
U1 = m1 * g * p1(2); 
U2 = m2 * g * p2(2);
U3 = m3 * g * p3(2);
U4 = m4 * g * p4(2);
U  = simplify( Ut + U1 + U2 + U3 + U4 );

% ----------- Stance Jacobian and Stance Jacobian Derivative --------------
% This is needed for dynamic equation with constraints 

% Stance Jacobian and its derivative
Jst_rhand = jacobian(p_rhand, q);
Jst_lhand = jacobian(p_lhand, q);
Jst_rfoot = jacobian(p_rfoot, q);
Jst_lfoot = jacobian(p_lfoot, q);

Jstdot_rhand = sym('Jdot', size(Jst_rhand));  
Jstdot_lhand = sym('Jdot', size(Jst_lhand));  
Jstdot_rfoot = sym('Jdot', size(Jst_rfoot));  
Jstdot_lfoot = sym('Jdot', size(Jst_lfoot));  


% COM Jacobian and its derivative
JpCOM = simplify(jacobian(pCOM, q));

dJpCOM = sym('dJpCOM', size(JpCOM));


% Compute the derivative of the Jacobian
for i = 1:size(Jstdot_rhand,1)
    for j = 1:size(Jstdot_rhand,2)
        temp_rhand = 0;
        temp_lhand = 0;
        temp_rfoot = 0;
        temp_lfoot = 0;
        temp_pCOM = 0;

        for k = 1:length(q)
            temp_rhand = temp_rhand + diff(Jst_rhand(i,j), q(k)) * dq(k);
            temp_lhand = temp_lhand + diff(Jst_lhand(i,j), q(k)) * dq(k);
            temp_rfoot = temp_rfoot + diff(Jst_rfoot(i,j), q(k)) * dq(k);
            temp_lfoot = temp_lfoot + diff(Jst_lfoot(i,j), q(k)) * dq(k);
            temp_pCOM = temp_pCOM + diff(JpCOM(i,j), q(k)) * dq(k);
        end

        Jstdot_rhand(i,j) = temp_rhand;
        Jstdot_lhand(i,j) = temp_lhand;
        Jstdot_rfoot(i,j) = temp_rfoot;
        Jstdot_lfoot(i,j) = temp_lfoot;
        dJpCOM(i,j) = temp_pCOM;
    end
end

Jstdot_rhand = simplify(Jstdot_rhand);  
Jstdot_lhand = simplify(Jstdot_lhand);  
Jstdot_rfoot = simplify(Jstdot_rfoot);  
Jstdot_lfoot = simplify(Jstdot_lfoot);  
dJpCOM = simplify(dJpCOM);


% ------------------------- Dynamics Equations ----------------------------
q_act = [q1; q2; q3; q4]; % Right arm, left arm, right leg, left leg angles actuated

% Flight dynamics (no contact)
[D, C, G, B] = LagrangianDynamics(T, U, q, dq, q_act);   

% =========================================================================
% ------------------------- Virtual Constraints ---------------------------
% =========================================================================

% Define Symbolic Variables for virual constraint
syms Kp Kd real
syms u1 u2 u3 u4 real
syms phid real
syms qd1 qd2 qd3 qd4 real
syms lambda_ra1 lambda_ra2 real
syms lambda_la1 lambda_la2 real
syms lambda_rl1 lambda_rl2 real
syms lambda_ll1 lambda_ll2 real

% Generalized input and contact force
u_sym = [u1; u2; u3; u4];
lambda_sym = [lambda_ra1; lambda_ra2; 
              lambda_la1; lambda_la2;
              lambda_rl1; lambda_rl2;
              lambda_ll1; lambda_ll2];

% Holonomic Virtual Constraint
h = [q1 - qd1; % Drive the right arm to a desired angle
     q2 - qd2; % Drive the left arm to a desired angle
     q3 - qd3; % Drive the right leg to a desired angle
     q4 - qd4]; % Drive the left leg to a desired angle

% Holonomic Jacobian
Jh = jacobian(h, q); %  Jh = dh/dq 
Jh = simplify(Jh);

Jhdotdq = jacobian(Jh * dq, q) * dq; % Jhdot * dq
Jhdotdq = simplify(Jhdotdq);

Jhdot = sym(zeros(size(Jh))); 
for i = 1:size(Jh,1)
    for j = 1:length(q)
        Jhdot(i,:) = Jhdot(i,:) + diff(Jh(i,:), q(j)) * dq(j);
    end
end

% First Derivative
Lfh = Jh * dq; % dh/dt
Lfh = simplify(Lfh);

% Useful for Second Derivative (not computed symbolically as its too intense) 
d2h__ = [jacobian(Jh*dq, q), Jh];

% =========================================================================
% --------------------------- Export Functions ----------------------------
% =========================================================================

if ~exist('./auto')
    mkdir('./auto')
end
addpath('./auto')

% Generate position files
matlabFunction(p_rhand, 'File', 'auto/auto_prhand');
matlabFunction(p_lhand, 'File', 'auto/auto_plhand');
matlabFunction(p_rfoot, 'File', 'auto/auto_prfoot');
matlabFunction(p_lfoot, 'File', 'auto/auto_plfoot');

% Generate Lagrangian Dynamics
matlabFunction(D, 'File', 'auto/auto_D');
matlabFunction(C, 'File', 'auto/auto_C');
matlabFunction(G, 'File', 'auto/auto_G');
matlabFunction(B, 'File', 'auto/auto_B');

% Generate stationary Jacobians and their derivatives
matlabFunction(Jst_rhand, 'File', 'auto/auto_Jst_rhand');
matlabFunction(Jst_lhand, 'File', 'auto/auto_Jst_lhand');
matlabFunction(Jst_rfoot, 'File', 'auto/auto_Jst_rfoot');
matlabFunction(Jst_lfoot, 'File', 'auto/auto_Jst_lfoot');

matlabFunction(Jstdot_rhand, 'File', 'auto/auto_Jstdot_rhand');
matlabFunction(Jstdot_lhand, 'File', 'auto/auto_Jstdot_lhand');
matlabFunction(Jstdot_rfoot, 'File', 'auto/auto_Jstdot_rfoot');
matlabFunction(Jstdot_lfoot, 'File', 'auto/auto_Jstdot_lfoot');

% Generate COM Jacobian and its derivative
matlabFunction(JpCOM, 'File', 'auto/auto_JpCOM');
matlabFunction(dJpCOM, 'File', 'auto/auto_dJpCOM');

% Generate corresponding virtual constraints for stance
matlabFunction(h, 'File', 'auto/auto_h');
matlabFunction(Jh, 'File', 'auto/auto_Jh');
matlabFunction(Jhdot, 'File', 'auto/auto_Jhdot');
matlabFunction(Jhdotdq, 'File', 'auto/auto_Jhdotdq');
matlabFunction(Lfh, 'File', 'auto/auto_Lfh');
matlabFunction(d2h__, 'File', 'auto/auto_d2h__');

