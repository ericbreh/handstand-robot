function [u, y, dy] = gait_control(x, params)

y = y_auto(x, params.a, params.b, params.c, params.d, params.th1_des);
dy = Lfy_auto(x, params.a, params.b, params.c, params.d, params.th1_des);

a = params.k_a;
eps = params.k_eps;

%v = -params.kp*y - params.kd*dy;
v = [psi(y(1), eps*dy(1), a)/(eps^2);
     psi(y(2), eps*dy(2), a)/(eps^2);
     psi(y(3), eps*dy(3), a)/(eps^2);
     psi(y(4), eps*dy(4), a)/(eps^2)];

u = LgLfy_auto(x, params.a, params.b, params.c, params.d, params.th1_des)\(-L2fy_auto(x, params.a, params.b, params.c, params.d, params.th1_des) + v);

end

function out = psi(in1, in2, a)

out = -sign(in2)*abs(in2)^a - sign(phi(in1, in2, a))*abs(phi(in1, in2, a))^(a/(2-a));

end

function out = phi(in1, in2, a)

out = in1 + sign(in2)*(abs(in2)^(2-a))/(2-a);

end