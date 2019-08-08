%% Regression on two-line intersection from images
clear all
close all
rng(0);
load('newLineData10ksmall.mat');
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
    imageInputLayer([inputSize], 'Normalization', 'zerocenter', 'Name', 'input')
    
    convolution2dLayer([5, 5], 32, 'Padding','same', 'Name', 'conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name','pool1')
    
    convolution2dLayer([3, 3], 32, 'Padding','same','Name', 'conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name','pool2')
    
    convolution2dLayer([3, 3], 32, 'Padding','same', 'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name','pool3')    
  
    fullyConnectedLayer(32, 'Name','fc1')
    depthConcatenationLayer(2,'Name','concat1')
    reluLayer('Name','relu4')
    
    fullyConnectedLayer(64, 'Name','fc2')
    reluLayer('Name','relu5')

    fullyConnectedLayer(2, 'Name', 'fc3')
    xyRegressionLayer('xyRegression')];

  
  
lgraph = layerGraph(layers);
fcskip = fullyConnectedLayer(32, 'Name','fcskip');
argmaxskip = newargmaxLayer('argmax', 1);
lgraph = addLayers(lgraph,argmaxskip);
lgraph = addLayers(lgraph,fcskip);
lgraph = connectLayers(lgraph,'relu1','argmax');
lgraph = connectLayers(lgraph,'argmax','fcskip');
lgraph = connectLayers(lgraph,'fcskip','concat1/in2');


%% Training parameters

miniBatchSize = 64; %size(trainIms, 4);
validationFreq = floor(length(trainLabels) / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',300, ...
    'InitialLearnRate',1e-3, ...
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
%    'GradientThresholdMethod', 'l2norm', ...
%    'GradientThreshold', 1e-5);
  
  
 %% Train network
[net, trainInfo] = trainNetwork(trainIms, trainLabels, lgraph, options);

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

%% Calculate angular error
rPred = [vPred(:, 1), 1 ./ vPred(:, 2)];
rPred(:, 3) = ones(length(rPred), 1);
rLabels = [squeeze(testLabels(1, 1, 1, :)), 1 ./ squeeze(testLabels(1, 1, 2, :))];
rLabels(:, 3) = ones(length(rPred), 1);

rollpred = atan(-rPred(:, 1) ./ sqrt(rPred(:, 2).^2 + rPred(:, 3).^2));
pitchpred = atan(rPred(:, 2) ./ rPred(:, 3));

roll = atan(-rLabels(:, 1) ./ sqrt(rLabels(:, 2).^2 + rLabels(:, 3).^2));
pitch = atan(rLabels(:, 2) ./ rLabels(:, 3));

figure
subplot(211)
histogram(rad2deg(rollpred - roll))
xlabel('Roll error [degrees]')
subplot(212)
histogram(rad2deg(pitchpred - pitch));
xlabel('Pitch error [degrees]')



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
