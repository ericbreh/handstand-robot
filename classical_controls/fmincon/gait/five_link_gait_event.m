function [event, isterminal, direction] = five_link_gait_event(~, x)

th1 = x(3) + x(7) - pi;

event = th1 - pi/8;
isterminal = 1;
direction = 1;

end