function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);


for i = 1:length(X)

    currX = X(:, i);
    nearest = [zeros(2, k); ones(1, k)*inf; zeros(1, k)];

    squared = (Xt - currX).^2;
    dists = squared(1, :) + squared(2, :);

    ksmallest = mink(dists, k);

    smallest_ind = find();
    % TODO DO SHIT HERE

    % for j = 1:length(Xt);
    %     dist = norm(currX - Xt(:, j));
    %     


    %     % for l = 1:k
    %     %     if dist < nearest(3, l)
    %     %         nearest(:, l) = [Xt(:, j); dist; Lt(j)];
    %     %         break;
    %     %     end
    %     % end

    % end

    labelsOut(i) = mode(nearest(4, :));

end

end

