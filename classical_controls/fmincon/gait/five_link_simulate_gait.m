function sim = five_link_simulate_gait(X, params)

x0 = X(params.x_idx);
params.a = X(params.a_idx);
params.b = X(params.b_idx);
params.c = X(params.c_idx);
params.d = X(params.d_idx);

options = odeset('Events', @five_link_gait_event);
[t, x] = ode45(@(t, x) five_link_gait_dynamics(t, x, params), [0 10], x0, options);

x_minus = x(end, :)';
x_plus = five_link_gait_impact_dynamics(x_minus, params);
dist = x_plus(1)-x0(1);
T_I = t(end);

n = length(t);
y = zeros(n, 4);
u = zeros(n, 4);
F_st = zeros(n, 2);
Fv_st = zeros(n, 1);
Fh_st = zeros(n, 1);

for i = 1:n
    [u(i, :), y(i, :), ~] = gait_control(x(i, :)', params);

    F_st(i, :) = F_st_auto(x(i, :)', u(i, :)')';
    Fv_st(i) = F_st(i, 2);
    Fh_st(i) = F_st(i, 1);
end

sim.t = t;
sim.x = x;

sim.x_plus = x_plus;
sim.x_minus = x_minus;

sim.u = u;
sim.y = y;

sim.F_st = F_st;
sim.Fv_st = Fv_st;
sim.Fh_st = Fh_st;

sim.dist = dist;
sim.T_I = T_I;

end