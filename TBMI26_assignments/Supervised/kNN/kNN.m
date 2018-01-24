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

    squared = (Xt - currX).^2;
    dists = sum(squared);

    [ksmallest, indices] = mink(dists, k);

    labelsOut(i) = mode(Lt(indices));
end

end

