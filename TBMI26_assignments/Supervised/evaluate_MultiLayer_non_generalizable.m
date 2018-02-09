%% This script will help you test out your single layer neural network code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 3; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );
numNeurons = length(unique(L));

%% Select a subset of the training features

numBins = 2; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

numTrainEx = 10;

Dsliced = Dt{1};
Dsliced = Dsliced(:, 1:numTrainEx);

Lsliced = Lt{1};
Lsliced = Lsliced(1:numTrainEx, :);

Xtrain = Xt{1};
Xtrain = Xtrain(:, 1:numTrainEx);

[h, w] = size(Xt{2});
[htrain, wtrain] = size(Xtrain);
% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};
%% Modify the X Matrices so that a bias is added

% The Training Data
Xtraining = [ones(1, wtrain); Xtrain];

% The Test Data
Xtest = [ones(1, w); Xt{2}];


%% Train your single layer network
% Note: You nned to modify trainSingleLayer() in order to train the network

numHidden = 10; % Change this, Number of hidde neurons 
numIterations = 4000; % Change this, Numner of iterations (Epochs)
learningRate = 0.002; % Change this, Your learningrate
[xh, xw] = size(Xtraining);
W0 = rand(numHidden, xh)*0.01; % Change this, Initiate your weight matrix W
V0 = rand(numNeurons, numHidden)*0.01; % Change this, Initiate your weight matrix V

%
tic
[W,V, trainingError, testError] = trainMultiLayer(Xtraining,Dsliced,Xtest,Dt{2}, W0,V0,numIterations, learningRate );
trainingTime = toc;
%% Plot errors
figure(1101)
clf
[mErr, mErrInd] = min(testError);
plot(trainingError,'k','linewidth',1.5)
hold on
plot(testError,'r','linewidth',1.5)
plot(mErrInd,mErr,'bo','linewidth',1.5)
hold off
title('Training and Test Errors, Multi-Layer')
legend('Training Error','Test Error','Min Test Error')

%% Calculate The Confusion Matrix and the Accuracy of the Evaluation Data
% Note: you have to modify the calcConfusionMatrix() function yourselfs.

[ Y, LMultiLayerTraining ] = runMultiLayer(Xtraining, W, V);
tic
[ Y, LMultiLayerTest ] = runMultiLayer(Xtest, W,V);
classificationTime = toc/length(Xtest);
% The confucionMatrix
cM = calcConfusionMatrix( LMultiLayerTest, Lt{2});

% The accuracy
acc = calcAccuracy(cM);

display(['Time spent training: ' num2str(trainingTime) ' sec'])
display(['Time spent calssifying 1 feature vector: ' num2str(classificationTime) ' sec'])
display(['Accuracy: ' num2str(acc)])

%% Plot classifications
% Note: You do not need to change this code.

if dataSetNr < 4
    plotResultMultiLayer(W,V,Xtraining,Lsliced,LMultiLayerTraining,Xtest,Lt{2},LMultiLayerTest)
else
    plotResultsOCR( Xtest, Lt{2}, LMultiLayerTest )
end
