%% Regression on two-line intersection from images
clear all
close all
rng(0);
load('vpData.mat');
addpath('../tools/')


dataSize = size(trainIms);
inputSize = dataSize(1:3);
squareSize = 5;

layers = [
    imageInputLayer([inputSize], 'Normalization', 'zerocenter', 'Name', 'input')
    
    convolution2dLayer([3, 3], 16, 'Padding','same', 'Name', 'conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name','pool1')
    
    convolution2dLayer([3, 3], 32, 'Padding','same','Name', 'conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name','pool2')
    
    convolution2dLayer([3, 3], 64, 'Padding','same', 'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    
    convolution2dLayer([3, 3], 64, 'Padding','same', 'Name','conv3')
    batchNormalizationLayer('Name','bn4')
    reluLayer('Name','relu4')
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name','pool3')  
    
    softmaxLayer
    indexLayer('index');
  
    fullyConnectedLayer(512, 'Name','fc1')
    reluLayer('Name','relu4')

    fullyConnectedLayer(3, 'Name', 'fc3')
    sphereLayer('sphere')
    crossProductRegressionLayer('cross')];


%% Training parameters

miniBatchSize = 64; %size(trainIms, 4);
validationFreq = floor(length(trainLabels) / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
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
   'ValidationPatience', 5);
  
  
 %% Train network
[net, trainInfo] = trainNetwork(trainIms, trainLabels, layers, options);

%% Prediction
vPredTrain = predict(net, trainIms, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');
vPred = predict(net, testIms, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');

vPredTrain = vPredTrain .* sign(vPredTrain(:, 3));
vPred = vPred .* sign(vPred(:, 3));

%% Calculate r^2 coefficients
vTrain = squeeze(trainLabels);
ybarTrain = mean(vTrain');
SStotTrain = sum((vTrain' - ybarTrain).^2);
SSresTrain = sum((vTrain' - vPredTrain).^2);
r2Train = 1 - SSresTrain ./ SStotTrain;

vTest = squeeze(testLabels);
ybarTest = mean(vTest');
SStotTest = sum((vTest' - ybarTest).^2);
SSresTest = sum((vTest' - vPred).^2);
r2Test = 1 - SSresTest ./ SStotTest;


%%
figure
subplot(231)
plot(squeeze(trainLabels(1, 1, 1, :)), vPredTrain(:, 1), '.')
legend(['Train: R2 = ', num2str(r2Train(1))])
xlabel('True Label'), ylabel('Predicted Label')
subplot(232)
plot(squeeze(trainLabels(1, 1, 2, :)), vPredTrain(:, 2), '.')
legend(['Train: R2 = ', num2str(r2Train(2))])
xlabel('True Label'), ylabel('Predicted Label')
subplot(233)
plot(squeeze(trainLabels(1, 1, 3, :)), vPredTrain(:, 3), '.')
xlabel('True Label'), ylabel('Predicted Label')
legend(['Train: R2 = ', num2str(r2Train(3))])

subplot(234)
plot(squeeze(testLabels(1, 1, 1, :)), vPred(:, 1), '.')
legend(['Test: R2 = ', num2str(r2Test(1))])
xlabel('True Label'), ylabel('Predicted Label')
subplot(235)
plot(squeeze(testLabels(1, 1, 2, :)), vPred(:, 2), '.')
legend(['Test: R2 = ', num2str(r2Test(2))])
xlabel('True Label'), ylabel('Predicted Label')
subplot(236)
plot(squeeze(testLabels(1, 1, 3, :)), vPred(:, 3), '.')
xlabel('True Label'), ylabel('Predicted Label')
legend(['Test: R2 = ', num2str(r2Test(3))])


%% Calculate angular error
% rPred = vPred(:, 1:2) ./ vPred(:, 3);
% rLabels = vTest(:, 1:2) ./ vTest(:, 3);

pitchpred= atan(-vPred(:, 3) ./ sqrt(vPred(:, 1).^2 + vPred(:, 2).^2));
rollpred = atan(vPred(:, 1) ./ vPred(:, 2));

vTest = vTest';
pitch = atan(-vTest(:, 3) ./ sqrt(vTest(:, 1).^2 + vTest(:, 2).^2));
roll = atan(vTest(:, 1) ./ vTest(:, 2));

figure
subplot(211)
ecdf(abs(rad2deg(pitchpred - pitch)))
xlabel('Pitch error [degrees]')
grid on
ylabel('CDF')
subplot(212)
ecdf(abs(rollpred - roll))
xlabel('Roll error [degrees]')
ylabel('CDF')
grid on



%% Calculate arcsin error
angErr = zeros(length(vPred), 1);
for i = 1:length(vPred)
  angErr(i) = rad2deg(asin(sqrt(sum(cross(vPred(i, :), vTest(i, :)).^2))));
end

figure
ecdf(angErr)
grid on
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
