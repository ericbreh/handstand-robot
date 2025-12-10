function xdot = five_link_swing_OL_dynamics(~, x, params)

u = swing_OL_control(x, params);

xdot = fdyn_su_auto(x) + gdyn_su_auto(x)*u;

end