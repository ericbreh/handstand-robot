clear; close all; clc;
restoredefaultpath;
addpath('./auto');
addpath('./gait_OL');
addpath('./plant_OL');
addpath('./stand_OL');
addpath('./swing_OL');

%% User Parameters

% Number of Steps
params.num_steps = 5;

% Control
params.k_a = 0.9;
params.k_eps = 0.1;

% State Indexes
params.x_idx = 1:14;
params.q_idx = 1:7;
params.qdot_idx = 8:14;

% Relabelling Matrix
params.R = eye(7);
params.R(3:4, 3:4) = [0 1; 1 0];

% Absolute Angle Transformation: th = T*q + d
params.T = [1 0 0 0 1;
            0 1 0 0 1;
            0 0 1 0 1;
            0 0 0 1 1;
            0 0 0 0 1];
params.d = [-pi; -pi; -pi; -pi; 0];

%% Simulation
% Set-up

prev_t_imp_idx = 0;
t0 = 0;

T = cell(params.num_steps+2, 1);
X = cell(params.num_steps+2, 1);
t_imp_idx = zeros(params.num_steps+2, 1);

% Gaits

% Desired limb angles
q3_des = pi - pi/12;
q4_des = pi + pi/12;
q5_des = pi/6;
th3_des = q5_des + q3_des - pi;
th4_des = q5_des + q4_des - pi;
th5_des = q5_des;
params.th_des = [0; 0; th3_des; th4_des; th5_des];

x0 = [0; 0.9239; 2.2253; 3.0107; q3_des; q4_des; q5_des;
      0.8653; 0.3584; -1.0957; -2.3078; 0.0000; 0.0000; 2.0323];

options = odeset('Events', @five_link_gait_OL_event);
for i = 1:params.num_steps
    [T{i}, X{i}] = ode45(@(t, x) five_link_gait_OL_dynamics(t, x, params), [t0 t0+10], x0, options);
    x_minus = X{i}(end, :)';
    x0 = five_link_gait_OL_impact_dynamics(x_minus, params);

    t_imp_idx(i) = height(T{i}) + prev_t_imp_idx;
    prev_t_imp_idx = t_imp_idx(i);
    t0 = T{i}(end);

end

x_gait = vertcat(X{1:i});
t_gait = vertcat(T{1:i});
for j = 1:height(x_gait)

    [u, y, dy] = gait_OL_control(x_gait(j, :)', params);
    u_gait(j, :) = u'; y_gait(j, :) = y'; dy_gait(j, :) = dy';
    
    F_st_gait(j, :) = F_st_auto(x_gait(j, :)', u_gait(j, :)')';

end

figure(1);
subplot(2, 1, 1);
plot(t_gait, y_gait(:,1), 'r', ...   % red
     t_gait, y_gait(:,2), 'g', ...   % green
     t_gait, y_gait(:,3), 'b', ...   % blue
     t_gait, y_gait(:,4), 'k');      % black
legend('y_1', 'y_2', 'y_3', 'y_4');

subplot(2, 1, 2);
plot(t_gait, F_st_gait(:, 1), 'r', ...
     t_gait, F_st_gait(:, 2), 'g');
legend("Horizontal Force", "Vertical Force");

%% Hand Plant
i = i + 1;
q_des = [pi - pi/2; pi + pi/2; pi - pi/2; 0; 2*pi/3];
params.th_des = params.T*q_des + params.d;

options = odeset('Events', @five_link_plant_OL_event);
[T{i}, X{i}] = ode45(@(t, x) five_link_plant_OL_dynamics(t, x, params), [t0 t0+10], x0, options);
x_minus = X{i}(end, :)';
x0 = five_link_plant_OL_impact_dynamics(x_minus, params);

t_imp_idx(i) = height(T{i}) + prev_t_imp_idx;
prev_t_imp_idx = t_imp_idx(i);
t0 = T{i}(end);

x_hp = X{i};
t_hp = T{i};

for j = 1:height(x_hp)

    [u, y, dy] = plant_OL_control(x_hp(j, :)', params);
    u_hp(j, :) = u'; y_hp(j, :) = y'; dy_hp(j, :) = dy';
    th_hp(j, :) = params.T*x_hp(j, 3:7)' + params.d;
    
    F_st_hp(j, :) = F_st_auto(x_hp(j, :)', u_hp(j, :)')';

end

figure(2);
subplot(2, 1, 1);
plot(t_hp, y_hp(:,1), 'r', ...   % red
     t_hp, y_hp(:,2), 'g', ...   % green
     t_hp, y_hp(:,3), 'b', ...   % blue
     t_hp, y_hp(:,4), 'k');      % black
legend('y_1', 'y_2', 'y_3', 'y_4');

subplot(2, 1, 2);
plot(t_hp, F_st_hp(:, 1), 'r', ...
     t_hp, F_st_hp(:, 2), 'g');
legend("Horizontal Force", "Vertical Force");

%% Swing up
i = i + 1;
q_des = [pi - pi/6; pi + pi/6; 0; 0; pi];
params.th_des = params.T*q_des + params.d;

options = odeset('Events', @five_link_swing_OL_event);
[T{i}, X{i}] = ode45(@(t, x) five_link_swing_OL_dynamics(t, x, params), [t0 t0+10], x0, options);
x_minus = X{i}(end, :)';
x0 = five_link_swing_OL_impact_dynamics(x_minus, params);

t_imp_idx(i) = height(T{i}) + prev_t_imp_idx;
prev_t_imp_idx = t_imp_idx(i);
t0 = T{i}(end);

x_su = X{i};
t_su = T{i};

for j = 1:height(x_su)

    [u, y, dy] = swing_OL_control(x_su(j, :)', params);
    u_su(j, :) = u'; y_su(j, :) = y'; dy_su(j, :) = dy';
    th_su(j, :) = params.T*x_su(j, 3:7)' + params.d;
    
    F_su(j, :) = F_su_auto(x_su(j, :)', u_su(j, :)')';

end

figure(3);
subplot(2, 1, 1);
plot(t_su, y_su(:,1), 'r', ...   % red
     t_su, y_su(:,2), 'g', ...   % green
     t_su, y_su(:,3), 'b', ...   % blue
     t_su, y_su(:,4), 'k');      % black
legend('y_1', 'y_2', 'y_3', 'y_4');

subplot(2, 1, 2);
plot(t_su, F_su(:, 1), 'r', ...
     t_su, F_su(:, 2), 'g');
legend("Horizontal Force", "Vertical Force");


%% Animate

x_tot = vertcat(X{:});
t_tot = vertcat(T{:});

q_tot = x_tot(:, params.q_idx);
qdot_tot = x_tot(:, params.qdot_idx);

figure(1000);
drawnow;
animateFiveLink(t_tot, q_tot);







