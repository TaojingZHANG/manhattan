% net = alexnet();
% %net.Layers
% 
% layersTransfer = net.Layers(1:end-3); %end-3
% layers = [layersTransfer; fullyConnectedLayer(2); xyRegressionLayer('xyRegression')];
% 
% inputSize = net.Layers(1).InputSize;

%% This network can overfit on the dataset
% inputSize = [240, 320];
% 
% layers = [
%     imageInputLayer([inputSize, 1])
%     
%     convolution2dLayer([5, 20], 8 ,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer([3, 10], 16, 'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer([2, 5], 32, 'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     fullyConnectedLayer(100)
%     reluLayer
%     dropoutLayer(0.5)
%     
%     fullyConnectedLayer(2)
%     xyRegressionLayer('xyRegression')];

%% Custom network

inputSize = [480, 640] / 2;

layers = [
    imageInputLayer([inputSize, 1], 'Normalization', 'none')
    
    convolution2dLayer([5   , 20], 64 ,'Padding','same')
    batchNormalizationLayer
    
    convolution2dLayer([5, 20], 64 ,'Padding','same')
    batchNormalizationLayer
    
    maxPooling2dLayer([1, 2],'Stride',2)
    
    convolution2dLayer([3, 10], 128, 'Padding','same')
    batchNormalizationLayer
    
    convolution2dLayer([3, 10], 128, 'Padding','same')
    batchNormalizationLayer
        
    maxPooling2dLayer([1, 2],'Stride',2)
    
    convolution2dLayer([2, 5], 256, 'Padding','same')
    batchNormalizationLayer

    maxPooling2dLayer([2, 2],'Stride',2)
    
%     convolution2dLayer([2, 5], 32, 'Padding','same')
%     reluLayer
%     
%     convolution2dLayer([2, 5], 16, 'Padding','same')
%     reluLayer
        
    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(2)
    xyRegressionLayer('xyRegression')];

%% Generate images and labels

load('YorkVpLabels');

%% Split training and validation

Ntrain = 70;
Ntest = length(vgtLabels) - Ntrain;
trainInds = randperm(length(vgtLabels), 70);
testInds = 1:length(vgtLabels);
testInds(trainInds) = 0;
testInds = find(testInds);
testInds = testInds(randperm(length(testInds)));

% Try with edge tranform
edgeData = zeros(size(imData));
for n = 1:size(imData, 4)
   edgeData(:, :, :, n) = edge(imData(:, :, :, n), 'canny');
end

% Normalize images
% imData = (imData - mean(imData, 4)) ./ ...
%     std(imData, [], 4);

scaleFac = 1 ./ std(vgtLabels');
trainLabels = vgtLabels(:, trainInds)' .* scaleFac;
testLabels = vgtLabels(:, testInds)' .* scaleFac;
% testLabels = scaleFac .* mod(testLabels ./ ...
%     sqrt(sum([testLabels, ones(length(testLabels), 1)].^2, 2)) + 0.5, 1) - 0.5;
trainIms = single(imresize(edgeData(:, :, :, trainInds), inputSize));
testIms = single(imresize(edgeData(:, :, :, testInds), inputSize));


%% Training parameters

miniBatchSize = 32;
validationFrequency = floor(numel(trainLabels)/miniBatchSize);

% Data augmentation
augmenter = imageDataAugmenter( ...
  'RandRotation',[-10, 10], ...
  'RandXReflection', false, ...
  'RandYReflection', false, ...
  'RandXTranslation', [-10, 10], ...
  'RandYTranslation', [-10, 10]);
% augimds = augmentedImageDatastore([480, 640], trainIms, 'DataAugmentation', ...
%   augmenter);

maxEpochs = 1;
initialLr = 1e-4;
lrDropFactor = 0.1;
lrDropPeriod = 200;
L2regularizer = 0.001;
verboseFrequency = 1;
sgdMomentum = 0.7;

options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',initialLr, ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle','every-epoch', ...
    'ValidationData',{testIms,testLabels}, ...
    'ValidationFrequency',validationFrequency, ...
    'VerboseFrequency',validationFrequency, ...
    'ValidationPatience', Inf, ...
    'Plots', 'none', ...
    'L2Regularization', L2regularizer, ...
    'Momentum', sgdMomentum, ...
    'LearnRateDropFactor', 1);  
  
 %% Train network
cellData = squeeze(num2cell(trainIms, 1:3));
cellTestData = squeeze(num2cell(testIms, 1:3));

nRandomEpochs = 1000;
lr = initialLr;
for e = 1:nRandomEpochs
  % Sample data
  epochImages = augment(augmenter, cellData);
  epochImages = cat(4, epochImages{:});
  epochLabels = zeros(length(trainLabels), 2);
 
  for i = 1:Ntrain
    temp = trainLabels(i, :)';
    transVp = augmenter.AffineTransforms(:, :, i)' * [temp; 1];
    epochLabels(i, :) = transVp(1:2);
  end
   
  epochTestImages = augment(augmenter, cellTestData);
  epochTestImages = cat(4, epochTestImages{:});
  epochTestLabels = zeros(length(testLabels), 2);
  for i = 1:Ntest
    temp = testLabels(i, :)';
    transVp = augmenter.AffineTransforms(:, :, i)' * [temp; 1];
    epochTestLabels(i, :) = transVp(1:2);
  end
  
  
  if e == 1
    [net, trainInfo] = trainNetwork(epochImages,epochLabels,layers,options);
  else
    [net, trainInfo] = trainNetwork(epochImages,epochLabels,net.Layers,options);
  end
  
  if mod(e, lrDropPeriod) == 0
      lr = lr * lrDropFactor;
  end
  if mod(e, verboseFrequency) == 0
      disp(e);
      vb = true;
  else
      vb = false;
  end
  
  options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate', lr, ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle','every-epoch', ...
    'ValidationData',{epochTestImages,epochTestLabels}, ...
    'ValidationFrequency',validationFrequency, ...
    'VerboseFrequency',validationFrequency, ...
    'ValidationPatience', Inf, ...
    'Plots', 'none', ...
    'L2Regularization', L2regularizer, ...
    'Verbose', vb, ...
    'Momentum', sgdMomentum, ...
    'LearnRateDropFactor', 1);

    
end

%% Prediction

vPred = predict(net,testIms);
predictionError = testLabels - vPred;


figure(1), 
subplot(211)
plot(testLabels(:, 1)), hold on, plot(vPred(:, 1), '--')
hold off
subplot(212)
plot(testLabels(:, 2)), hold on, plot(vPred(:, 2), '--')
hold off

%% Training fit
% Sample data
epochImages = augment(augmenter, cellData);
epochImages = cat(4, epochImages{:});
epochLabels = zeros(length(trainLabels), 2);
for i = 1:Ntrain
    temp = trainLabels(i, :)';
    transVp = augmenter.AffineTransforms(:, :, i)' * [temp; 1];
    epochLabels(i, :) = transVp(1:2);
end

vPredTrain = predict(net,epochImages);
predictionErrorTrain = epochLabels - vPredTrain;


figure(2), 
subplot(211)
plot(epochLabels(:, 1)), hold on, plot(vPredTrain(:, 1), '--')
hold off
subplot(212)
plot(epochLabels(:, 2)), hold on, plot(vPredTrain(:, 2), '--')
hold off

%% Augmented test data fit

% Sample data
epochTestImages = augment(augmenter, cellTestData);
epochTestImages = cat(4, epochTestImages{:});
epochTestLabels = zeros(length(testLabels), 2);
for i = 1:Ntest
    temp = testLabels(i, :)';
    transVp = augmenter.AffineTransforms(:, :, i)' * [temp; 1];
    epochTestLabels(i, :) = transVp(1:2);
end
  

vPredAug = predict(net,epochTestImages);
predictionErrorAug = epochTestLabels - vPredAug;


figure(3), 
subplot(211)
plot(epochTestLabels(:, 1)), hold on, plot(vPredAug(:, 1), '--')
hold off
subplot(212)
plot(epochTestLabels(:, 2)), hold on, plot(vPredAug(:, 2), '--')
hold off

%% Augmented pitch roll fit
truePitchRoll = zeros(size(testLabels));
estPitchRoll = zeros(size(testLabels));
for i = 1:length(epochTestLabels)
    tmp = augmenter.AffineTransforms(:, :, i)' \ [epochTestLabels(i, :), 1]';
    tmp(1:2) = tmp(1:2) ./ scaleFac';
    tmp = augmenter.AffineTransforms(:, :, i)' * tmp;
    tmpNorm = tmp / norm(tmp);
    truePitch = atan(-tmpNorm(1) ...
        ./ sqrt(tmpNorm(2).^2 + tmpNorm(3).^2));
    trueRoll = mod(atan(tmpNorm(2) ./ tmpNorm(3)), pi) - pi / 2;
    truePitchRoll(i, :) = [truePitch, trueRoll];
    
    tmp = augmenter.AffineTransforms(:, :, i)' \ [vPredAug(i, :), 1]';
    tmp(1:2) = tmp(1:2) ./ scaleFac';
    tmp = augmenter.AffineTransforms(:, :, i)' * tmp;
    tmpNorm = tmp / norm(tmp);
    estPitch = atan(-tmpNorm(1) ...
        ./ sqrt(tmpNorm(2).^2 + tmpNorm(3).^2));
    estRoll = mod(atan(tmpNorm(2) ./ tmpNorm(3)), pi) - pi / 2;
    estPitchRoll(i, :) = [estPitch, estRoll];
end

figure(4), 
subplot(211)
plot(rad2deg(truePitchRoll(:, 1))), hold on, plot(rad2deg(estPitchRoll(:, 1)), '--')
xlabel('Pitch')
legend('True', 'Estimated')
hold off
subplot(212)
plot(rad2deg(truePitchRoll(:, 2))), hold on, plot(rad2deg(estPitchRoll(:, 2)), '--')
xlabel('Roll')
legend('True', 'Estimated')
hold off
