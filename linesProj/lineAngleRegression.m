%% Euler Angel (Pitch, Roll) regression on Aachen dataset
clear all
close all
rng(0);
load('lineAnglesData');
addpath('../tools/')

%N = 1000;
% trainIms = trainIms(:, :, :, 1:N);
% trainLabels = trainLabels(:, :, :, 1:N);
% testIms = testIms(:, :, :, 1:0.1*N);
% testLabels = testLabels(:, :, :, 1:0.1*N);

%% Custom network

dataSize = size(trainIms);
inputSize = dataSize(1:3);
squareSize = 5;

layers = [
    imageInputLayer([inputSize], 'Normalization', 'zerocenter')
    
    convolution2dLayer([squareSize, squareSize], 2 * squareSize - 2 ,'Padding','same', 'Stride', 2)
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([3, 3], 16 ,'Padding','same', 'Stride', 2)
    batchNormalizationLayer
    maxPooling2dLayer([2, 2])
    reluLayer
    
    convolution2dLayer([3, 3], 32 ,'Padding','same', 'Stride', 2)
    batchNormalizationLayer
    maxPooling2dLayer([2, 2])
    reluLayer
    
    fullyConnectedLayer(32)
    reluLayer
    
    fullyConnectedLayer(1)
    regressionLayer];

%% Specify convolutional weights

layers(2).Weights = zeros(squareSize, squareSize, 1, 2*squareSize - 2);
layers(2).Weights(:, :, 1, 1) = 0.1 * diag(ones(1, squareSize)) / sqrt(squareSize);
rot = 1 / (squareSize - 1) * 90;
for i = 1:2*squareSize - 3
  layers(2).Weights(:, :, 1, i + 1) = ...
    imrotate(layers(2).Weights(:, :, 1, 1), -i * rot, 'bilinear', 'crop');
end

layers(2).WeightLearnRateFactor = 0;
layers(2).WeightL2Factor = 0;

%% Training parameters

miniBatchSize = 32;
aachenDsTrain.MiniBatchSize = miniBatchSize;
validationFreq = floor(length(trainLabels) / miniBatchSize);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'L2Regularization', 0.001, ...
    'VerboseFrequency', 10, ...
    'ValidationData', {testIms, testLabels}, ...
   'ValidationFrequency', validationFreq, ...
   'ValidationPatience', 20);
  
  
 %% Train network
[net, trainInfo] = trainNetwork(trainIms, trainLabels, layers, options);

%% Prediction
vPredTrain = predict(net, trainIms, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');
vPred = predict(net, testIms, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');

%% Calculate r^2 coefficients
trainAngles = squeeze(trainLabels);
ybarTrain = mean(trainAngles);
SStotTrain = sum((trainAngles - ybarTrain).^2);
SSresTrain = sum((trainAngles - vPredTrain).^2);
r2Train = 1 - SSresTrain ./ SStotTrain;

testAngles = squeeze(testLabels);
ybarTest = mean(testAngles);
SStotTest = sum((testAngles - ybarTest).^2);
SSresTest = sum((testAngles - vPred).^2);
r2Test = 1 - SSresTest ./ SStotTest;

%% Error in angle

figure
subplot(211)
plot(squeeze(trainLabels), vPredTrain, '.')
legend(['Train: R2 = ', num2str(r2Train)])
subplot(212)
plot(squeeze(testLabels), vPred, '.')
legend(['Test: R2 = ', num2str(r2Test)])


%%
layer = 2;
name = net.Layers(layer).Name;

channels = 1:2*squareSize - 2;
I = deepDreamImage(net,layer,channels,'PyramidLevels',1);

figure
for i = 1:8
    subplot(3,3,i)
    imshow(I(:,:,:,i))
end
