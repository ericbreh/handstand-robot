function x_plus = five_link_swing_OL_impact_dynamics(x_minus, params)

q_plus = x_minus(params.q_idx);
qdot_plus = qdot_su_i_plus_auto(x_minus);

x_plus = [q_plus; qdot_plus];

end