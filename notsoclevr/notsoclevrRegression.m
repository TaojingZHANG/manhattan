clear all
close all
rng(0);
addpath('../tools/')

%% Generate notsoclevr
[trainIms, testIms, trainLabels, testLabels] = notsoclevrGen('uniform');

%% Custom network

dataSize = size(trainIms);
inputSize = dataSize(1:3);

% layers = [
%     imageInputLayer(inputSize, 'Normalization', 'none')
%     
%     coordConvLayer('');
%     convolution2dLayer([5, 5], 32, 'Padding','same', 'Stride', 2)
%     batchNormalizationLayer
%     reluLayer
% 
%     %coordConvLayer('');
%     convolution2dLayer([3, 3], 64 ,'Padding','same', 'Stride', 2)
%     batchNormalizationLayer
%     reluLayer
%     
%     %coordConvLayer('');
%     convolution2dLayer([3, 3], 128 ,'Padding','same', 'Stride', 2)
%     batchNormalizationLayer
%     reluLayer
%   
%     fullyConnectedLayer(512)
%     reluLayer
% 
%     fullyConnectedLayer(2)
%     xyRegressionLayer('intersection regression')]; ...

layers = [
    imageInputLayer(inputSize, 'Normalization', 'none')
    
    coordConvLayer('CoordConv');
    convolution2dLayer([1, 1], 8, 'Padding','same')
    reluLayer

    convolution2dLayer([1, 1], 8 ,'Padding','same')
    reluLayer
    
    convolution2dLayer([1, 1], 8 ,'Padding','same')
    reluLayer

    
    convolution2dLayer([3, 3], 8 ,'Padding','same')
    reluLayer
      
    convolution2dLayer([3, 3], 2 ,'Padding','same')
    reluLayer

    maxPooling2dLayer(64, 'stride', 64)
    xyRegressionLayer('intersection regression')];



%% Training parameters

miniBatchSize = 32;
validationFreq = floor(length(trainLabels) / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',5e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'L2Regularization', 5e-4, ...
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
    imshow(I(:,:,1,i))
end
