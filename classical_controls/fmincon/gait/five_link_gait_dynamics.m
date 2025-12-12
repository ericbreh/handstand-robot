function xdot = five_link_gait_dynamics(~, x, params)

[u, ~, ~] = gait_control(x, params);

xdot = fdyn_auto(x) + gdyn_auto(x)*u;

end