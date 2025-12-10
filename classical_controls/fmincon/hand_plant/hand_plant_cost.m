function J = handplantcost(X, params)

sim = five_link_simulate_hand_plant(X, params);

u2 = vecnorm(sim.u, 2, 2).^2;
J = trapz(sim.t, u2)/sim.dist;

end