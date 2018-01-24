function [cM] = calcConfusionMatrix(Lclass, Ltrue)
classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

% Add your own code here

for i = 1:numClasses
    for j = 1:numClasses
        cM(i, j) = sum((Ltrue == i).*(Lclass == j));
    end
end

end

