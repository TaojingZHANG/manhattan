%% Euler Angel (Pitch, Roll) regression on Aachen dataset
clear all
close all
rng(0);
load('lineData.mat');
addpath('../tools/')


%% Custom network

dataSize = size(trainIms);
inputSize = dataSize(1:3);

layers = [
    imageInputLayer([inputSize], 'Normalization', 'none')
    
    convolution2dLayer([3, 3], 8 ,'Padding','same', 'Stride', 1)
    reluLayer
    
    convolution2dLayer([3, 3], 16 ,'Padding','same', 'Stride', 1)
    reluLayer
    
    convolution2dLayer([3, 3], 32, 'Padding','same', 'Stride', 1)
    reluLayer
    
    convolution2dLayer([3, 3], 64, 'Padding','same', 'Stride', 1)
    reluLayer
    
    %averagePooling2dLayer([2, 2])
        
    fullyConnectedLayer(32)
    reluLayer
    
    fullyConnectedLayer(2)
    xyRegressionLayer('xyRegression')];


%% Training parameters

miniBatchSize = 16;
aachenDsTrain.MiniBatchSize = miniBatchSize;
validationFreq = floor(length(trainLabels) / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'Plots','none', ...
    'L2Regularization', 0, ...
    'VerboseFrequency', 10, ...
    'ValidationData', {testIms, testLabels}, ...
   'ValidationFrequency', validationFreq, ...
   'ValidationPatience', 20);
  
  
 %% Train network
[net, trainInfo] = trainNetwork(trainIms, trainLabels, layers, options);
