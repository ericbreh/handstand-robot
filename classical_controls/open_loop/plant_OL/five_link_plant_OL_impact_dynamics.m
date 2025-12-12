function x_plus = five_link_plant_OL_impact_dynamics(x_minus, params)

q_plus = params.R*x_minus(params.q_idx);
qdot_plus = params.R*qdot_hp_plus_auto(x_minus);

x_plus = [q_plus; qdot_plus];

end