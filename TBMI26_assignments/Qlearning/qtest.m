while 1
	gwinit(world);
	s = gwstate();

	while ~s.isterminal
		action = chooseaction(Q, s.pos(1), s.pos(2), ...
			[1 2 3 4], [1 1 1 1], 0);
		s = gwaction(action);
		pause(0.1)
		gwdraw
		gwplotallarrows(Q)
	end
end
