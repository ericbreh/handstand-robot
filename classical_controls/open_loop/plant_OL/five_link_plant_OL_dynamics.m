function xdot = five_link_plant_OL_dynamics(~, x, params)

u = plant_OL_control(x, params);

xdot = fdyn_st_auto(x) + gdyn_st_auto(x)*u;

end