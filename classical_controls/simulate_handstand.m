% -------------------------------------------------------------------------
% simulate_handstand.m
% -------------------------------------------------------------------------
% This is a complete, event-driven simulation of a handstand. 
% -------------------------------------------------------------------------
% Last Modified: 07-November-2025
% -------------------------------------------------------------------------

function simulate_handstand()
    % ---------------------------------------------------------------------
    % If the auto folder is not generated, you need to generate it first.
    % To do so, you need to run: generate_files.m
    % This should create and populate an auto folder. This only ever needs
    % to be done once (Unless you change the dynamics in the
    % generate_files.m, then you will need to regenerate it).
    % ---------------------------------------------------------------------

    clear all; close all; clc;

    % =====================================================================
    % ------------ 0.  Add the path for the different folders -------------
    % Note: you don't need to run this everytime, only at the start 
    % (unless you change folders). Uncomment the code first time you run.
    % =====================================================================
    
    addpath('./auto/');
    addpath('./events/'); 
    addpath('./dynamics/');
    addpath('./animate/'); 

    % =====================================================================
    % ------------- 1. Define the simulation parameters -------------------
    % =====================================================================
    
    % Model Parameters
    % Torso
    params.mt = 0.02; % Change if necessary
    params.wt = 0.4;  % Change if necessary
    params.lt = 0.6;  % Change if necessary
    params.It = (1/12) * params.mt * (params.lt^2 + params.wt^2); % Assume torso is a rectangle
    params.dtx = 0.0; % Change if necessary
    params.dty = 0.0; % Change if necessary

    % Right arm
    params.m1 = 8;    % Change if necessary
    params.l1 = 0.8;  % Change if necessary
    params.d1 = params.l1/2;
    params.I1 = (1/12) * params.m1 * params.l1^2; % Assume arms are rods
    
    % Left arm  (Assume Left arm = Right arm)
    params.m2 = params.m1;
    params.l2 = params.l1;
    params.d2 = params.l2/2;
    params.I2 = (1/12) * params.m2 * params.l2^2; % Assume arms are rods

    % Right leg
    params.m3 = 10;   % Change if necessary
    params.l3 = 1.0;  % Change if necessary
    params.d3 = params.l3/2;
    params.I3 = (1/12) * params.m3 * params.l3^2; % Assume legs are rods
    
    % Left arm  (Assume Left leg = Right leg)
    params.m4 = params.m3;
    params.l4 = params.l3;
    params.d4 = params.l4/2;
    params.I4 = (1/12) * params.m4 * params.l4^2; % Assume legs are rods
    
    % Environment
    params.g  = 9.81;
    params.mu = 0.8;

    % Trajectory Angles
    params.qd1 = deg2rad(30);  % Change
    params.qd2 = params.qd1;
    params.qd3 = deg2rad(40);  % Change
    params.qd4 = params.qd3;

    % Check the 5-link system
    q = [0; 1.5; 0; deg2rad(30); deg2rad(30); deg2rad(40); deg2rad(40)];
    plot_5link(q, params)
    
end