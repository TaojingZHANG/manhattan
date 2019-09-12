%% Horizon line regression
clear all
close all
rng(0);

%% Load database
addpath('../tools/')
load('horizonDs.mat');

%% Custom network

inputSize = [227, 227, 3];

% layers = [
%     imageInputLayer([inputSize], 'Normalization', 'none', 'Name', 'input')
%     
%     convolution2dLayer([7, 7], 64, 'Padding','same', 'Stride', 2, 'Name', 'conv1')
%     batchNormalizationLayer('Name','bn1')
%     reluLayer('Name','relu1')
%     
%     convolution2dLayer([5, 5], 32, 'Padding','same','Name', 'conv2')
%     batchNormalizationLayer('Name','bn2')
%     reluLayer('Name','relu2')
%         
%     convolution2dLayer([5, 5], 16, 'Padding','same', 'Name','conv3')
%     batchNormalizationLayer('Name','bn3')
%     reluLayer('Name','relu3')
%     
%     softargmaxLayer2('softargmax')
%   
%     fullyConnectedLayer(32, 'Name','fc1')
%     reluLayer('Name','relu4')
% 
%     fullyConnectedLayer(3, 'Name', 'fc3')
%     sphereLayer('sphere')
%     crossProductRegressionLayer('cross')];

transfernet = importCaffeNetwork('deploy_alexnet_places365.prototxt','alexnet_places365.caffemodel');
layersTransfer = transfernet.Layers(2:end-3);
layers = [
  imageInputLayer(inputSize, 'Normalization', 'none');
  layersTransfer
  fullyConnectedLayer(3, 'WeightLearnRateFactor', 10,'BiasLearnRateFactor', 10)
  sphereLayer('Sphere')
  crossProductRegressionLayer('cross')];


% layers(2).WeightLearnRateFactor = 0;
% layers(2).BiasLearnRateFactor = 0;
% 
% layers(6).WeightLearnRateFactor = 0;
% layers(6).BiasLearnRateFactor = 0;
% 
% layers(10).WeightLearnRateFactor = 0;
% layers(10).BiasLearnRateFactor = 0;
% 
% layers(12).WeightLearnRateFactor = 0;
% layers(12).BiasLearnRateFactor = 0;
% 
% layers(14).WeightLearnRateFactor = 0;
% layers(14).BiasLearnRateFactor = 0;

%% Training parameters

miniBatchSize = 64;
L2reg = 0;
lr = 1e-4;
lrDropRate = 0.1;
lrDropPeriod = 3;
validationFreq = 100;
maxEpochs = 10;

horizonDsTrain.MiniBatchSize = miniBatchSize;

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',lr, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',lrDropRate, ...
    'LearnRateDropPeriod',lrDropPeriod, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'L2Regularization', L2reg, ...
    'VerboseFrequency', 10, ...
    'ValidationData', horizonDsTest, ...
   'ValidationFrequency', validationFreq, ...
   'ValidationPatience', 300);
  
  
 %% Train network
[net, trainInfo] = trainNetwork(horizonDsTrain, layers, options);

%% Make prediction on validation data

pred = predict(net, horizonDsTest, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');

%% Evaluate error based on horizon error

err = zeros(length(pred), 1);
labels = zeros(size(pred));
for n = 1:length(pred)
  labels(n, :) = squeeze(horizonDsTest.Labels{n})';
  err(n, :) = sum(cross(labels(n, :), pred(n, :)).^2);
end

%% 

horizonDir = '../wildhorizon/';
fileName = 'metadata.csv';

fid = fopen([horizonDir, fileName]);
imdata = textscan(fid, '%s %f %f %f %f %*[^\n]', 'Delimiter', ',');
fid = fclose(fid);


fid = fopen([horizonDir, 'split/test.txt']);
test = textscan(fid, '%s %*[^\n]');
fid = fclose(fid);

test = {(test{1}(1:k:end))};

horErr = zeros(length(pred), 1);
for n = 1:length(pred)
  lHat = pred(n, :);
  lTrue = labels(n, :);
  name = test{1}{n};
  index = find(contains(imdata{1}, name));
  i = imfinfo([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  x1 = 0; x2 = i.Width;
  
  y1 = (-lTrue(1) * x1 / xScale - lTrue(3) * 1) / lTrue(2) * yScale;
  y2 = (-lTrue(1) * x2 / xScale- lTrue(3) * 1) / lTrue(2) * yScale;
  y1Hat = (-lHat(1) * x1 / xScale - lHat(3) * 1) / lHat(2) * yScale;
  y2Hat = (-lHat(1) * x2 / xScale - lHat(3) * 1) / lHat(2) * yScale;
  horErr(n) = max(abs([y1Hat - y1, y2Hat - y2])) / i.Height;
end
  
%% Calculate empirical cumulative error distribution

figure
auc = calc_auc(horErr, true, '', false);


%%

% protofile = '~/deephorizon/models/regression/init_best_so_huber/solver.proto';
% datafile =  '~/deephorizon/models/regression/init_best_so_huber/init_best_so_huber.caffemodel';
% 
% net = importCaffeNetwork(protofile,datafile)

%% Visualize first conv layer using deep dream 

layer = 2;
name = net.Layers(layer).Name;

channels = 1:64;
I = deepDreamImage(net,layer,channels,'PyramidLevels',1);

figure
for i = 1:length(channels)
  subplot(8,8,i)
  imagesc(I(:, :, :, i))
end

%% Show samples for some images

for n = randperm(length(pred))
  lHat = pred(n, :);
  lTrue = labels(n, :);
  name = test{1}{n};
  index = find(contains(imdata{1}, name));
  i = imfinfo([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  I = imread([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  x1 = 0; x2 = i.Width;
  
  y1 = (-lTrue(1) * x1 / xScale - lTrue(3) * 1) / lTrue(2) * yScale;
  y2 = (-lTrue(1) * x2 / xScale- lTrue(3) * 1) / lTrue(2) * yScale;
  y1Hat = (-lHat(1) * x1 / xScale - lHat(3) * 1) / lHat(2) * yScale;
  y2Hat = (-lHat(1) * x2 / xScale - lHat(3) * 1) / lHat(2) * yScale;
  
  figure(1); clf;
  sz = size(I); sz = sz(1:2);
  figure(1); clf;
  image(I, 'XData', [1 sz(2)] - (sz(2)+1)/2, 'YData', [sz(1) 1] - (sz(1)+1)/2)
  axis xy image off
  hold on
  plot([x1 x2] - sz(2) / 2, [y1, y2], 'b', 'LineWidth', 3);
  plot([x1 x2] - sz(2) / 2, [y1Hat, y2Hat], 'b--', 'LineWidth', 3);
  hold off
  title(['Error = ', num2str(horErr(n))])
  pause
  
end

