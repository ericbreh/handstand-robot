function xdot = five_link_gait_OL_dynamics(~, x, params)

u = gait_OL_control(x, params);

xdot = fdyn_st_auto(x) + gdyn_st_auto(x)*u;

end