function [event, isterminal, direction] = five_link_swing_OL_event(~, x)

q3 = x(5);

event = q3 - pi/6;
isterminal = 1;
direction = 1;

end