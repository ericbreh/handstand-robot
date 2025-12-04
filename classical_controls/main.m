clear; close all; clc;
addpath('./auto');
addpath('./animate');
addpath('./dynamics');
addpath('./events');
addpath('./fmincon');

%% Optimizer
clear; close all; clc;
x0_guess = [0; 0.9239; 2.2253; 3.0107; deg2rad(90), deg2rad(270), 0.5236;
      0.8653; 0.3584; -1.0957; -2.3078; 0.0000; 0.0000; 2.0323];

a0 = [pi/6; 0; 0; 0];
b0 = [pi/6; 0; 0; 0];
c0 = [pi/6; 0; 0; 0];
d0 = [0; 0; 0; 0];

options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'interior-point', ...
    'MaxFunctionEvaluations', 1e5, ...
    'MaxIterations', 5000);

[Xopt, Jopt] = fmincon(@cost, [x0_guess; a0, b0, c0, d0], [], [], [], [], [], [], @nonlcon. options);

%%
x0 = Xopt(1:14);

