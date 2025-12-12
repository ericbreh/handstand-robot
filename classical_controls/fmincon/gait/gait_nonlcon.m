function [c, ceq] = gait_nonlcon(X, params)

sim = five_link_simulate_gait(X, params);

x0 = X(params.x_idx);
x_plus = sim.x_plus;

% Equality constraint (periodic)
ceq = x0(2:end)-x_plus(2:end);

% Inequality constraints
c_uni = -min(sim.Fv_st);
c_friction = max(abs(sim.Fh_st./sim.Fv_st)) - params.mu;
c_speed = params.vel - (sim.dist/sim.T_I);

c = [c_uni, c_friction, c_speed];

end