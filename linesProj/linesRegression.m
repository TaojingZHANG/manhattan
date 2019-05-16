%% Regression on two-line intersection from images
clear all
close all
rng(0);
load('lineData.mat');
addpath('../tools/')


% %% Custom network
% 
dataSize = size(trainIms);
inputSize = dataSize(1:3);
squareSize = 5;
% 
layers = [
    imageInputLayer([inputSize], 'Normalization', 'zerocenter')
%     
%     convolution2dLayer([squareSize, squareSize], 2 * squareSize - 2,'Padding','same', 'Stride', 2)
%     batchNormalizationLayer
%     reluLayer
    
    convolution2dLayer([5, 5], 64 ,'Padding','same', 'Stride', 2)
    batchNormalizationLayer
    maxPooling2dLayer([2, 2])
    reluLayer
    
    convolution2dLayer([3, 3], 128 ,'Padding','same', 'Stride', 2)
    batchNormalizationLayer
    maxPooling2dLayer([2, 2])
    reluLayer
    
    convolution2dLayer([3, 3], 256 ,'Padding','same', 'Stride', 2)
    batchNormalizationLayer
    maxPooling2dLayer([2, 2])
    reluLayer
    
  
    %dropoutLayer(0.5);
%     fullyConnectedLayer(32)
%     reluLayer

    fullyConnectedLayer(2)
    %twoLineLayer('two lines')
    xyRegressionLayer('intersection regression')];
% 
% %% Specify convolutional weights

% layers(2).Weights = zeros(squareSize, squareSize, 1, 2*squareSize - 2);
% layers(2).Weights(:, :, 1, 1) = 0.1 * diag(ones(1, squareSize)) / sqrt(squareSize);
% rot = 1 / (squareSize - 1) * 90;
% for i = 1:2*squareSize - 3
%   layers(2).Weights(:, :, 1, i + 1) = ...
%     imrotate(layers(2).Weights(:, :, 1, 1), -i * rot, 'bilinear', 'crop');
% end
% 
% layers(2).WeightLearnRateFactor = 0;
% layers(2).WeightL2Factor = 0;

%% Retrain angle network
% load('angleNet.mat')
% layers = angleNet.Layers;
% layers = layers(1:8);
% 
% layers(2).WeightLearnRateFactor = 0;
% layers(2).WeightL2Factor = 0;
% 
% % layers(5).WeightLearnRateFactor = 0;
% % layers(5).WeightL2Factor = 0;
% % 
% % layers(9).WeightLearnRateFactor = 0;
% % layers(9).WeightL2Factor = 0;
% 
% layers = [layers
%   convolution2dLayer([3, 3], 128 ,'Padding','same', 'Stride', 2)
%   batchNormalizationLayer
%   maxPooling2dLayer([2, 2])
%   reluLayer
%   
%   convolution2dLayer([3, 3], 256 ,'Padding','same', 'Stride', 1)
%   batchNormalizationLayer
%   maxPooling2dLayer([2, 2])
%   reluLayer
%   
%   %dropoutLayer(0.5)
%   fullyConnectedLayer(2048)
%   reluLayer
%   
%   %dropoutLayer(0.5)
%   fullyConnectedLayer(32)
%   reluLayer
%   
%   fullyConnectedLayer(2)
%   xyRegressionLayer('xyregression')];


%% Training parameters

miniBatchSize = 32;
validationFreq = floor(length(trainLabels) / miniBatchSize);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'L2Regularization', 0.01, ...
    'VerboseFrequency', 10, ...
    'ValidationData', {testIms, testLabels}, ...
   'ValidationFrequency', validationFreq, ...
   'ValidationPatience', Inf);
  
  
 %% Train network
[net, trainInfo] = trainNetwork(trainIms, trainLabels, layers, options);

%% Prediction
vPredTrain = predict(net, trainIms, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');
vPred = predict(net, testIms, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');

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
    imshow(I(:,:,:,i))
end
