function gwplotallarrows(Q)
% Plot all arrows using gwplotarrow and the Q matrix.
  
global GWXSIZE;
global GWYSIZE;
global GWTERM;

% Arrow directions
% Change this to select arrow directions from the Q matrix.
A = ones(GWYSIZE, GWXSIZE);

for x = 1:GWXSIZE
    for y = 1:GWYSIZE
        if ~GWTERM(x,y)
            gwplotarrow([x y], chooseaction(Q, x, y, ...
				[1 2 3 4], [1 1 1 1], 0));
        end
    end
end

drawnow;
hold on;

    



