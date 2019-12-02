%% Horizon line regression
clear all
close all
rng(0);

%% Load database
addpath('../tools/');
addpath('../horizonProj/');
load('binDs.mat');

%% Custom network

inputSize = [224, 224, 3];

Nclasses = 100;

% Replicate Resnet
net = layerGraph(resnet50);
net = replaceLayer(net, 'input_1', imageInputLayer(inputSize, 'Normalization', 'none', 'name', 'input'));
net = removeLayers(net, 'fc1000'); %fullyConnectedLayer(Nclasses, 'Name', 'fc'));
net = removeLayers(net, 'fc1000_softmax'); %softmaxLayer('Name', 'softmax'));
net = removeLayers(net, 'ClassificationLayer_fc1000');

% Add fully connected layer
net = addLayers(net, fullyConnectedLayer(2 * Nclasses, 'Name', 'fc'));
net = connectLayers(net, 'avg_pool', 'fc');

% use Nclasses sigmoid and binary cross entropy layers
net = addLayers(net, sigmoidLayer('sig'));
net = connectLayers(net, 'fc', 'sig');

net = addLayers(net, binaryCrossEntropyLayer('be'));
net = connectLayers(net, 'sig', 'be');


%% Training parameters

miniBatchSize = 32;
L2reg = 0;
lr = 1e-3;  
lrDropRate = 1; 
lrDropPeriod = 10;
validationFreq = 300;
maxEpochs = 50;

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
    'ValidationData', binDsVal, ...
   'ValidationFrequency', validationFreq, ...
   'ValidationPatience', 300);
  
  
 %% Train network
binDsTrain.randomCrop = true;
binDsTrain.horizontalFlip = true;
[net, trainInfo] = trainNetwork(binDsTrain, net, options);

%% Make prediction on validation data

binDsTest.randomCrop = false;
binDsTest.horizontalFlip = false;
pred = predict(net, binDsVal, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');

%% Arg max prediction
rho = pred(:, 1:Nclasses);
theta = pred(:, Nclasses+1:end);

[~, rhomax] = max(rho, [], 2);
[~, thetamax] = max(theta, [], 2);

rhoEst = binDsTest.rhoClasses(rhomax);
thetaEst = binDsTest.thetaClasses(thetamax);

%%

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
cropSize = 224;
rhoTrue = zeros(length(rhoEst), 1);
thetaTrue = zeros(length(thetaEst), 1);
for n = 1:length(rhoEst)

  name = test{1}{n};
  index = find(contains(imdata{1}, name));
  i = imfinfo([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  
  x1 = imdata{2}(index);
  y1 = imdata{3}(index);
  x2 = imdata{4}(index);
  y2 = imdata{5}(index);
  M = [x1, y1, 1; x2, y2, 1];
  l = null(M);
  
  rhoTrue(n) = (x2 * y1 - y2 * x1) / sqrt( (y2-y1)^2 + (x2 - x1)^2) / scale;
  thetaTrue(n) = atand((y2-y1) / (x2 - x1));
  
  scale = min([i.Height, i.Width]) / cropSize;
  
  x0Hat = 0;
  y0Hat = rhoEst(n) * scale / cosd(abs(thetaEst(n)));
  
  x1Hat = rhoEst(n) * scale / sind(thetaEst(n));
  y1Hat = 0;

  Mhat = [x0Hat, y0Hat, 1; x1Hat, y1Hat, 1];
  lhat = null(Mhat);
  lhat(1) = -lhat(1);
  
  x1 = -i.Width / 2; x2 = i.Width / 2;
  y1Im = -(l(1) * x1 + l(3)) / l(2);
  y2Im = -(l(1) * x2 + l(3)) / l(2);
  
  y1ImHat = -(lhat(1) * x1 + lhat(3)) / lhat(2);
  y2ImHat = -(lhat(1) * x2 + lhat(3)) / lhat(2);
  
  horErr(n) = max(abs([y1ImHat - y1Im, y2ImHat - y2Im])) / i.Height;
  ysave(n, :) = [y1Im, y2Im] / i.Height;
  yHatsave(n, :) = [y1ImHat, y2ImHat] / i.Height;
end
  
%% Calculate empirical cumulative error distribution

figure
auc = calc_auc(horErr, true, '', false);

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
  
  scale = min([i.Height, i.Width]) / cropSize;
  
  x0Hat = 0;
  y0Hat = rhoEst(n) * scale / cosd(abs(thetaEst(n)));
  
  x1Hat = rhoEst(n) * scale / sind(thetaEst(n));
  y1Hat = 0;
  
% %   rhoTrue= (x2 * y1 - y2 * x1) / sqrt( (y2-y1)^2 + (x2 - x1)^2) / scale;
% %   thetaTrue = atand((y2-y1) / (x2 - x1));
% %    
% %   x0True = 0;
% %   y0True = rhoTrue * scale / cosd(abs(thetaTrue));
% %   
% %   x1True = rhoTrue * scale / sind(thetaTrue);
% %   y1True = 0;
% % 
% %   Mtrue = [x0True, y0True, 1; x1True, y1True, 1];
% %   lTrue = [-tand(thetaTrue); -1; rhoTrue / cosd(abs(thetaTrue))];
% %   lTrue = lTrue / lTrue(end);
  
  Mhat = [x0Hat, y0Hat, 1; x1Hat, y1Hat, 1];
  lhat = null(Mhat);
  lhat(1) = -lhat(1);
  
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

