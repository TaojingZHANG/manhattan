%% Euler Angel (Pitch, Roll) regression on Aachen dataset
clear all
close all
rng(0);

%% Load database
load('horizonDs.mat');

%% Custom network

inputSize = [227, 227, 3];

transfernet = alexnet;
layersTransfer = transfernet.Layers(2:end-3);
layers = [
  imageInputLayer(inputSize, 'Normalization', 'none');
  layersTransfer
  fullyConnectedLayer(3, 'WeightLearnRateFactor', 10,'BiasLearnRateFactor', 10)
  sphereLayer('Sphere')
  crossProductRegressionLayer('cross')];


%% Training parameters

miniBatchSize = 64;
L2reg = 0;
lr = 1e-3;
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

err = zeros(size(pred));
labels = zeros(size(pred));
for n = 1:length(pred)
  labels(n, :) = squeeze(horizonDsTest.Labels{n})';
  err(n, :) = squeeze(horizonDsTest.Labels{n})' - pred(n, :);
end


%% 

horizonDir = '/media/sf_axelsVirtualBox/wildhorizon/';
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
  
  y1 = (-lTrue(1) * x1 - lTrue(3) * 1) / lTrue(2);
  y2 = (-lTrue(1) * x2 - lTrue(3) * 1) / lTrue(2);
  y1Hat = (-lHat(1) * x1 - lHat(3) * 1) / lHat(2);
  y2Hat = (-lHat(1) * x2 - lHat(3) * 1) / lHat(2);
  horErr(n) = max(abs([y1Hat - y1, y2Hat - y2])) / i.Height;
end
  
  
%% Show samples for some images

for n = 1:10
  lHat = pred(n, :);
  lTrue = labels(n, :);
  name = test{1}{n};
  index = find(contains(imdata{1}, name));
  i = imfinfo([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  I = imread([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  x1 = 0; x2 = i.Width;
  
  y1 = (-lTrue(1) * x1 - lTrue(3) * 1) / lTrue(2);
  y2 = (-lTrue(1) * x2 - lTrue(3) * 1) / lTrue(2);
  y1Hat = (-lHat(1) * x1 - lHat(3) * 1) / lHat(2);
  y2Hat = (-lHat(1) * x2 - lHat(3) * 1) / lHat(2);
  
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


%%

% protofile = '~/deephorizon/models/regression/init_best_so_huber/solver.proto';
% datafile =  '~/deephorizon/models/regression/init_best_so_huber/init_best_so_huber.caffemodel';
% 
% net = importCaffeNetwork(protofile,datafile)
