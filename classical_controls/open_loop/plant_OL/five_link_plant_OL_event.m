function [event, isterminal, direction] = five_link_plant_OL_event(~, x)

p_hp = p_hp_auto(x);

event = p_hp(2);
isterminal = 1;
direction = -1;

end