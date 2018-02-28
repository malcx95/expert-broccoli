numEpisodes = 5000;
world = 1;
startEpsilon = 0.9;
alpha = 0.1;
gamma = 0.99;

gwinit(world);

s = gwstate()

Q = rand(s.xsize, s.ysize, 4);

epsilon = startEpsilon;

for episode=1:numEpisodes
	
	epsilon = startEpsilon - (episode/numEpisodes)*startEpsilon;
	
	gwinit(world);

	s = gwstate();
	
	while ~s.isterminal
		action = chooseaction(Q, s.pos(1), s.pos(2), ...
			[1 2 3 4], [1 1 1 1], epsilon);
		new_s = gwaction(action);
		Q(s.pos(1), s.pos(2), action) = (1-alpha)*Q(s.pos(1), s.pos(2), action) + ...
					alpha*(s.feedback + gamma*max(Q(new_s.pos(1), new_s.pos(2), :)));
		s = new_s;
	end
	episode
end

