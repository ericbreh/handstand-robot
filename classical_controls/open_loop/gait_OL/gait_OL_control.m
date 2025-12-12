function [u, y, dy] = gait_OL_control(x, params)

y = y_st_auto(x, params.th_des);
dy = Lfy_st_auto(x, params.th_des);

a = params.k_a;
eps = params.k_eps;

v = [psi(y(1), eps*dy(1), a)/(eps^2);
     psi(y(2), eps*dy(2), a)/(eps^2);
     psi(y(3), eps*dy(3), a)/(eps^2);
     psi(y(4), eps*dy(4), a)/(eps^2)];

u = LgLfy_st_auto(x, params.th_des)\(-L2fy_st_auto(x, params.th_des) + v);

end

function out = psi(in1, in2, a)

out = -sign(in2)*abs(in2)^a - sign(phi(in1, in2, a))*abs(phi(in1, in2, a))^(a/(2-a));

end

function out = phi(in1, in2, a)

out = in1 + sign(in2)*(abs(in2)^(2-a))/(2-a);

end