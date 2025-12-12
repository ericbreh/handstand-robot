function plot_5link_auto(q, params)
% PLOT_5LINK_AUTO Plot 5-link handstand using generated symbolic functions
% q = [x; y; phi; q1; q2; q3; q4]
% params = struct with any extra plotting parameters

% Torso rectangle (center = [x;y], angle = phi)
x = q(1); y = q(2); phi = q(3);
q1 = q(4); q2 = q(5); q3 = q(6); q4 = q(7);

% Load positions from generated files
prhand = auto_prhand(params.l1,params.lt,phi,q1,params.wt,x,y);
plhand = auto_plhand(params.l2,params.lt,phi,q2,params.wt,x,y);
prfoot = auto_prfoot(params.l3,params.lt,phi,q3,params.wt,x,y);
plfoot = auto_plfoot(params.l4,params.lt,phi,q4,params.wt,x,y);

% Rectangle corners in torso frame
corners = [-params.wt/2 -params.lt/2; params.wt/2 -params.lt/2; params.wt/2 params.lt/2; -params.wt/2 params.lt/2]';
R = [cos(phi) -sin(phi); sin(phi) cos(phi)];
torso_rot = R * corners + [x; y];

% Plot torso
fill(torso_rot(1,:), torso_rot(2,:), [0.8 0.8 1], 'FaceAlpha',0.5)
hold on

% Shoulder and hip positions (approximated from torso rectangle)
shoulder_r = torso_rot(:,4); % top right
shoulder_l = torso_rot(:,3); % top left
hip_r      = torso_rot(:,1); % bottom right
hip_l      = torso_rot(:,2); % bottom left

% Plot sticks (arms in red, legs in blue)
plot([shoulder_r(1) prhand(1)], [shoulder_r(2) prhand(2)], 'r-', 'LineWidth',2)
plot([shoulder_l(1) plhand(1)], [shoulder_l(2) plhand(2)], 'r-', 'LineWidth',2)
plot([hip_r(1) prfoot(1)], [hip_r(2) prfoot(2)], 'b-', 'LineWidth',2)
plot([hip_l(1) plfoot(1)], [hip_l(2) plfoot(2)], 'b-', 'LineWidth',2)

% Plot joints
plot(x, y, 'ko', 'MarkerFaceColor','k', 'MarkerSize',6)
plot([shoulder_r(1), shoulder_l(1), hip_r(1), hip_l(1)], ...
     [shoulder_r(2), shoulder_l(2), hip_r(2), hip_l(2)], ...
     'ko', 'MarkerFaceColor','k', 'MarkerSize',6)

axis equal
grid on
xlabel('X [m]')
ylabel('Y [m]')
title('5-Link Handstand Model')
end
