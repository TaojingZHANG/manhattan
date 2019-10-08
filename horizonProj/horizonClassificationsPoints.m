%% Horizon line regression
clear all
close all
rng(0);

%% Load database
addpath('../tools/');
load('horizonBinsDs.mat');


%% Custom network

inputSize = [224, 224, 3];

Nclasses = length(centroids);
%transfernet = importCaffeNetwork('deploy_alexnet_places365.prototxt','alexnet_places365.caffemodel');
% layersTransfer = transfernet.Layers(2:end-3);
% layers = [
%   imageInputLayer(inputSize, 'Normalization', 'none', 'name', 'input');
%   layersTransfer
%   fullyConnectedLayer(Nclasses, 'Name', 'fc')
%  	softmaxLayer
%   classificationLayer];

net = layerGraph(resnet50);
net = replaceLayer(net, 'input_1', imageInputLayer(inputSize, 'Normalization', 'none', 'name', 'input'));
net = replaceLayer(net, 'fc1000', fullyConnectedLayer(Nclasses, 'Name', 'fc'));
net = replaceLayer(net, 'fc1000_softmax', softmaxLayer('Name', 'softmax'));
net = replaceLayer(net, 'ClassificationLayer_fc1000', classificationLayer('Name', 'output'));



% layers(17).WeightLearnRateFactor = 10;
% layers(17).BiasLearnRateFactor = 10;
% 
% layers(20).WeightLearnRateFactor = 10;
% layers(20).BiasLearnRateFactor = 10;
% 
% layers(23).WeightLearnRateFactor = 10;
% layers(23).BiasLearnRateFactor = 10;

%% Training parameters

miniBatchSize = 32;
L2reg = 0.005;
lr = 1e-4;  
lrDropRate = 0.1; 
lrDropPeriod = 3;
validationFreq = 300;
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
[net, trainInfo] = trainNetwork(horizonDsTrain, net, options);

%% Make prediction on validation data

pred = predict(net, horizonDsTest, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');

%%
labels = zeros(length(pred), 1);
for n = 1:length(pred)
  labels(n) = double(horizonDsTest.Labels{n})';
end

horizonDir = '../wildhorizon/';
fileName = 'metadata.csv';

fid = fopen([horizonDir, fileName]);
imdata = textscan(fid, '%s %f %f %f %f %*[^\n]', 'Delimiter', ',');
fid = fclose(fid);


fid = fopen([horizonDir, 'split/test.txt']);
test = textscan(fid, '%s %*[^\n]');
fid = fclose(fid);

test = {(test{1}(1:end))};

horErr = zeros(length(pred), 1);
ysave = zeros(length(pred), 2);
yHatsave = zeros(length(pred), 2);
[~, argmax] = max(pred');
weighted = pred * centroids;
nWeights = 50;
for n = 1:length(pred)

  name = test{1}{n};
  index = find(contains(imdata{1}, name));
  i = imfinfo([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  
  x1 = imdata{2}(index);
  y1 = imdata{3}(index);
  x2 = imdata{4}(index);
  y2 = imdata{5}(index);
  M = [x1, y1, 1; x2, y2, 1];
  l = null(M);
  
%   y1Hat = centroids(argmax(n), 1) * yScale;
%   y2Hat = centroids(argmax(n), 1) * yScale;
%   y1Hat = weighted(n, 1) * yScale;
%   y2Hat = weighted(n, 2) * yScale;

  [w, inds] = sort(pred(n, :), 'descend');
  inds = inds(1:nWeights);
  w = w(1:nWeights) / sum(w(1:nWeights));

%   inds = labels(n); % ground truth bin
%   w = 1;
  
  y1Hat = w * centroids(inds, 1) * yScale;
  y2Hat = w * centroids(inds, 2) * yScale;


  Mhat = [x1, y1Hat, 1; x2, y2Hat, 1];
  lhat = null(Mhat);
  
  x1 = -i.Width / 2; x2 = i.Width / 2;
  y1Im = -(l(1) * x1 + l(3)) / l(2);
  y2Im = -(l(1) * x2 + l(3)) / l(2);
  
  y1ImHat = -(lhat(1) * x1 + lhat(3)) / lhat(2);
  y2ImHat = -(lhat(1) * x2 + lhat(3)) / lhat(2);
  
  horErr(n) = max(abs([y1ImHat - y1Im, y1ImHat - y2Im])) / i.Height;
  ysave(n, :) = [y1Im, y2Im] / i.Height;
  yHatsave(n, :) = [y1ImHat, y2ImHat] / i.Height;
end
  
%% Calculate empirical cumulative error distribution

figure
auc = calc_auc(horErr, true, '', false);


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
  name = test{1}{n};
  index = find(contains(imdata{1}, name));
  i = imfinfo([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  I = imread([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  
  x1 = imdata{2}(index);
  y1 = imdata{3}(index);
  x2 = imdata{4}(index);
  y2 = imdata{5}(index);
  M = [x1, y1, 1; x2, y2, 1];
  l = null(M);
  
  y1Hat = centroids(argmax(n), 1) * yScale;
  y2Hat = centroids(argmax(n), 1) * yScale;

  Mhat = [x1, y1Hat, 1; x2, y2Hat, 1];
  lhat = null(Mhat);
  
  x1 = -i.Width / 2; x2 = i.Width / 2;
  y1Im = -(l(1) * x1 + l(3)) / l(2);
  y2Im = -(l(1) * x2 + l(3)) / l(2);
  
  y1ImHat = -(lhat(1) * x1 + lhat(3)) / lhat(2);
  y2ImHat = -(lhat(1) * x2 + lhat(3)) / lhat(2);
  
  figure(1); clf;
  sz = size(I); sz = sz(1:2);
  figure(1); clf;
  image(I, 'XData', [1 sz(2)] - (sz(2)+1)/2, 'YData', [sz(1) 1] - (sz(1)+1)/2)
  axis xy image off
  hold on
  plot([x1 x2], [y1Im, y2Im], 'b', 'LineWidth', 3);
  plot([x1 x2], [y1ImHat, y2ImHat], 'b--', 'LineWidth', 3);
  hold off
  title(['Error = ', num2str(horErr(n))])
  pause
  
end

