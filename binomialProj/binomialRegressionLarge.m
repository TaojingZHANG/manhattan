%% Horizon line regression
clear all
close all
rng(0);

%% Load database
addpath('../tools/');
addpath('../horizonProj/');
addpath('../sordProj/');
load('binDs_large.mat');

%% Custom network

inputSize = [224, 224, 3];

Nclasses = 100;

% Replicate Resnet
load('resnet50');
net = layerGraph(resnet);
net = replaceLayer(net, 'input_1', imageInputLayer(inputSize, 'Normalization', 'none', 'name', 'input'));
net = removeLayers(net, 'fc1000'); 
net = removeLayers(net, 'fc1000_softmax');
net = removeLayers(net, 'ClassificationLayer_fc1000');

% Add fully connected layer with 10 times the learning rate
net = addLayers(net, fullyConnectedLayer(2 * Nclasses, 'Name', 'fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10));
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
lrDropRate = 0.1; 
lrDropPeriod = 10;
validationFreq = 300;
maxEpochs = 50;
gradientMomentum = 0.9;
binDsTrain.MiniBatchSize = miniBatchSize;

    
binDsTrain.randomCrop = true;
binDsTrain.horizontalFlip = true;

%% Custom training loop

validationLoss = zeros(maxEpochs, 1);
for e = 1:maxEpochs
    
    
    options = trainingOptions('sgdm', ...
        'Momentum', gradientMomentum, ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',1, ...
        'InitialLearnRate',lr, ...
        'LearnRateSchedule','none', ...
        'Shuffle','every-epoch', ...
        'Plots','none', ...
        'L2Regularization', L2reg, ...
        'VerboseFrequency', 100, ...
        'ValidationData', binDsVal, ...
        'ValidationFrequency', validationFreq, ...
        'ValidationPatience', Inf);
    
    if e == 1
    	[net, trainInfo] = trainNetwork(binDsTrain, net, options);
    else
       [net, trainInfo] = trainNetwork(binDsTrain, layerGraph(net), options);
    end

    validationLoss(e) = trainInfo.ValidationLoss(end);
    
    if e > 3
        if validationLoss(e) > validationLoss(e-3)
            lr = lr * lrDropRate;
        end
    end
    
    if mod(e, 5) == 0
        name = ['checkpoints/bin_', date, '_', num2str(e)];
        save(name, 'net', 'trainInfo', 'validationLoss')
    end
    disp(e);
end


%% Make prediction on validation data

binDsTest.randomCrop = false;
binDsTest.horizontalFlip = false;
pred = predict(net, binDsTest);

%% Prediction

predRho = pred(:, 1:Nclasses);
predTheta = pred(:, Nclasses+1:end);
rhoEst = zeros(length(pred), 1);
thetaEst = zeros(length(pred), 1);
avCount = zeros(Nclasses, 1);
L = floor(Nclasses / 4);
for i = 1:length(pred)
  countsRho = zeros(Nclasses, 1);
  countsTheta = zeros(Nclasses, 1);
  for j = 1:Nclasses
    inds = mod(j + (-L:L), Nclasses) + 1;
    countsRho(j) = sum(predRho(i, inds));
    countsTheta(j) = sum(predTheta(i, inds));
  end
  [~, amRho] = max(countsRho);
  [~, amTheta] = max(countsTheta);
  rhoEst(i) = binDsTest.rhoClasses(amRho);
  thetaEst(i) = binDsTest.thetaClasses(amTheta);
end

%%

horizonDir = '../wildhorizon_large/';
fileName = 'metadata.csv';

fid = fopen([horizonDir, fileName]);
imdata = textscan(fid, '%s %f %f %f %f %f %f %*[^\n]', 'Delimiter', ',');
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
  
  height = imdata{2}(index);
  width = imdata{3}(index);
  x1 = imdata{4}(index);
  y1 = imdata{5}(index);
  x2 = imdata{6}(index);
  y2 = imdata{7}(index);
  scale = min([height, width]) / cropSize;
  M = [x1, y1, 1; x2, y2, 1];
  l = null(M);
    
  rhoTrue(n) = (x2 * y1 - y2 * x1) / sqrt( (y2-y1)^2 + (x2 - x1)^2) / scale;
  thetaTrue(n) = atand((y2-y1) / (x2 - x1));
    
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
  
  horErr(n) = max(abs([y1ImHat - y1Im, y2ImHat - y2Im])) / height;
  ysave(n, :) = [y1Im, y2Im] / height;
  yHatsave(n, :) = [y1ImHat, y2ImHat] /height;
end
  
%% Calculate empirical cumulative error distribution

figure
auc = calc_auc(horErr, true, '', false);

%% Show samples for some images

count = 1;
for n = randperm(length(pred))
  name = test{1}{n};
  index = find(contains(imdata{1}, name));
  i = imfinfo([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  I = imread([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
  
  height = imdata{2}(index);
  width = imdata{3}(index);
  x1 = imdata{4}(index);
  y1 = imdata{5}(index);
  x2 = imdata{6}(index);
  y2 = imdata{7}(index);
  scale = min([height, width]) / cropSize;
  M = [x1, y1, 1; x2, y2, 1];
  l = null(M);
    
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
  
  m = mod(count,4) + 1;
  sz = size(I); sz = sz(1:2);
  figure(1);
  subplot(2,2,m)
  image(I, 'XData', [1 sz(2)] - (sz(2)+1)/2, 'YData', [sz(1) 1] - (sz(1)+1)/2)
  axis xy image off
  hold on
  plot([x1 x2], [y1Im, y2Im], 'b', 'LineWidth', 3);
  plot([x1 x2], [y1ImHat, y2ImHat], 'b--', 'LineWidth', 3);
  hold off
  %title(['Error = ', num2str(horErr(n))])
  pause
  
  count = count + 1;
end

