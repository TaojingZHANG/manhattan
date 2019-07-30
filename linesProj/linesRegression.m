%% Regression on two-line intersection from images
clear all
close all
rng(0);
load('lineData.mat');
addpath('../tools/')

useCoordConv = false;
useTransposeTraining = false;

if ~useCoordConv
  trainIms = trainIms(:, :, 1, :);
  testIms = testIms(:, :, 1, :);
end

if useTransposeTraining
  trainImsOld = trainIms;
  testImsOld = testIms;
  for n = size(trainIms, 4)
    trainIms(:, :, 1, n) = trainImsOld(:, :, 1, n) - trainImsOld(:, :, 1, n)';
  end
   for n = size(testIms, 4)
    testIms(:, :, 1, n) = testImsOld(:, :, 1, n) - testImsOld(:, :, 1, n)';
  end
end


% %% Custom network
% 
dataSize = size(trainIms);
inputSize = dataSize(1:3);
squareSize = 5;

layers = [
    imageInputLayer([inputSize], 'Normalization', 'zerocenter')
    
    convolution2dLayer([3, 3], 8, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([3, 3], 16, 'Padding','same')
   batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([3, 3], 32, 'Padding','same')
   batchNormalizationLayer
    reluLayer

    convolution2dLayer([3, 3], 64 ,'Padding','same')
   batchNormalizationLayer
    reluLayer
    
%     convolution2dLayer([3, 3], 128 ,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
    
    newargmaxLayer('argmax', 0.5);
  
    fullyConnectedLayer(1024)
    reluLayer
    
%     fullyConnectedLayer(64)
%     reluLayer
%     
%     fullyConnectedLayer(32)
%     reluLayer

    fullyConnectedLayer(2)
    %twoLineLayer('two lines')
    xyRegressionLayer('intersection regression')]; ...
    %sphericalRegressionLayer('Spherical Regression', 1e-10)];


%% Training parameters

miniBatchSize = 8;
validationFreq = floor(length(trainLabels) / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',1, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'L2Regularization', 0, ...
    'VerboseFrequency', 10, ...
    'ValidationData', {testIms, testLabels}, ...
   'ValidationFrequency', validationFreq, ...
   'ValidationPatience', Inf);
  
  
 %% Train network
[net, trainInfo] = trainNetwork(trainIms, trainLabels, layers, options);

%% Prediction
vPredTrain = predict(net, trainIms, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');
vPred = predict(net, testIms, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');

% vPred = vPred ./ sqrt(sum(vPred.^2, 2));
% vPredTrain = vPredTrain ./ sqrt(sum(vPredTrain.^2, 2));
% 
% testLabels = testLabels ./ sum(testLabels.^2, 3);
% trainLabels = trainLabels ./ sum(trainLabels.^2, 3);

%% Calculate r^2 coefficients
trainAngles = squeeze(trainLabels);
ybarTrain = mean(trainAngles');
SStotTrain = sum((trainAngles' - ybarTrain).^2);
SSresTrain = sum((trainAngles' - vPredTrain).^2);
r2Train = 1 - SSresTrain ./ SStotTrain;

testAngles = squeeze(testLabels);
ybarTest = mean(testAngles');
SStotTest = sum((testAngles' - ybarTest).^2);
SSresTest = sum((testAngles' - vPred).^2);
r2Test = 1 - SSresTest ./ SStotTest;


%%
figure
subplot(221)
plot(squeeze(trainLabels(1, 1, 1, :)), vPredTrain(:, 1), '.')
legend(['Train: R2 = ', num2str(r2Train(1))])
subplot(222)
plot(squeeze(trainLabels(1, 1, 2, :)), vPredTrain(:, 2), '.')
legend(['Train: R2 = ', num2str(r2Train(2))])

subplot(223)
plot(squeeze(testLabels(1, 1, 1, :)), vPred(:, 1), '.')
legend(['Test: R2 = ', num2str(r2Test(1))])
subplot(224)
plot(squeeze(testLabels(1, 1, 2, :)), vPred(:, 2), '.')
legend(['Test: R2 = ', num2str(r2Test(2))])

%%
layer = 2;
name = net.Layers(layer).Name;

channels = 1:8;
I = deepDreamImage(net,layer,channels,'PyramidLevels',1);

figure
for i = 1:8
    subplot(3,3,i)
    imshow(I(:,:,1,i))
end
