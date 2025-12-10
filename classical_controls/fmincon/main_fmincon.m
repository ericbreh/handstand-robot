clear; close all; clc;
addpath('./auto');
addpath('./animate');
addpath('./dynamics');
addpath('./events');

params.th1_des = pi/8;
params.k_p = 1;
params.k_d = 0.1;

params.k_a = 0.9;
params.k_eps = 0.1;

params.mu = 0.75;
params.vel = 0.7;

params.x_idx = 1:14;
params.q_idx = 1:7;
params.qdot_idx = 8:14;

params.a_idx = 15:18;
params.b_idx = 19:22;
params.c_idx = 23:26;
params.d_idx = 27:30;

params.R = eye(7);
params.R(3:4, 3:4) = [0 1; 1 0];
params.R(5:6, 5:6) = [0 1; 1 0];

%% Gait Optimizer
x0_guess = [0; 0.9239; 2.2253; 3.0107; deg2rad(135); deg2rad(225); 0.5236;
      0.8653; 0.3584; -1.0957; -2.3078; 0.0000; 0.0000; 2.0323];

a0 = [pi/6; 0; 0; 0];
b0 = [pi/3; 0; 0; 0];
c0 = [0; 0; 0; 0];
d0 = [0; 0; 0; 0];

options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'interior-point', ...
    'MaxFunctionEvaluations', 1e5, ...
    'MaxIterations', 5000);

%sim = five_link_simulate_gait([x0_guess; a0; b0; c0; d0], params);

[Xopt, Jopt] = fmincon(@(x) gait_cost(x, params), [x0_guess; a0; b0; c0; d0], [], [], [], [], [], [], @(x) gait_nonlcon(x, params), options);

%%
x0 = Xopt(1:14);

