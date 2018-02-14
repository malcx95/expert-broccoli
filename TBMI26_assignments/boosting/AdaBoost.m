% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces);
nonfaces = double(nonfaces);

% figure(1);
% colormap gray;
% for k=1:25
%     subplot(5,5,k), imagesc(faces(:,:,10*k));
%     axis image;
%     axis off;
% end
% 
% figure(2);
% colormap gray;
% for k=1:25
%     subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
%     axis image;
%     axis off;
% end

% Generate Haar feature masks
nbrHaarFeatures = 30;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

% figure(3);
% colormap gray;
% for k = 1:25
%     subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
%     axis image;
%     axis off;
% end

% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 1000; 
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

nbrBaseClassifiers = 50;

[N, M] = size(xTrain);
D = ones(1, M)/M;

K = ones(1, nbrBaseClassifiers);
T = ones(1, nbrBaseClassifiers);
P = ones(1, nbrBaseClassifiers);
Alpha = ones(1, nbrBaseClassifiers);

for i = 1:nbrBaseClassifiers
    i
    [kmin, tau, p] = Brute(xTrain, D, yTrain);
    K(i) = kmin;
    T(i) = tau;
    P(i) = p;
    C = WeakClassifier(tau, p, xTrain(kmin, :));
    e = WeakClassifierError(C, D, yTrain);
    alpha = 0.5 * log((1 - e)/e);
    Alpha(i) = alpha;
    D = D.*exp(-alpha*yTrain.*C);
    D = D/sum(D);
end

%% Extract test data

nbrTestExamples = 1000; 

testImages  = cat(3,faces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)),...
                    nonfaces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

%% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.

testaccs = [];
trainaccs = [];

for j = 1:nbrBaseClassifiers

    Htest = StrongClassifier(xTest, Alpha(1:j), T(1:j), P(1:j), K(1:j));
    Htrain = StrongClassifier(xTrain, Alpha(1:j), T(1:j), P(1:j), K(1:j));

    testacc = sum(Htest == yTest)/(2*nbrTestExamples);
    trainacc = sum(Htrain == yTrain)/(2*nbrTrainExamples);

    testaccs = [testaccs, testacc];
    trainaccs = [trainaccs, trainacc];

end

figure(1111)

plot([1:nbrBaseClassifiers], testaccs, '-r', [1:nbrBaseClassifiers], trainaccs, '-g');
legend('Test accuracy', 'Train accuracy')
title('Da accs')

%% Plot the error of the strong classifier as  function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.


