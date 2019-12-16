%% Horizon line regression
clear all
close all
rng(0);

%% Load database
addpath('../tools/');
addpath('../horizonProj/');
load('sordDs_large.mat');

sordDsTrain.rhoScale = 1;
sordDsVal.rhoScale = 1;
sordDsTest.rhoScale = 1;

sordDsTrain.thetaScale = 1;
sordDsVal.thetaScale = 1;
sordDsTest.thetaScale = 1;

%% Custom network

inputSize = [224, 224, 3];

Nclasses = 100;

% Replicate Resnet
load('resnet50');
net = layerGraph(resnet);
net = replaceLayer(net, 'input_1', imageInputLayer(inputSize, 'Normalization', 'none', 'name', 'input'));
net = removeLayers(net, 'fc1000'); %fullyConnectedLayer(Nclasses, 'Name', 'fc'));
net = removeLayers(net, 'fc1000_softmax'); %softmaxLayer('Name', 'softmax'));
net = removeLayers(net, 'ClassificationLayer_fc1000');

% Add 2 fully connected layers, one for rho and one for theta
% Use 10 times higher learning rate on fully connected layers
net = addLayers(net, fullyConnectedLayer(Nclasses, 'Name', 'fc1', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10));
net = addLayers(net, fullyConnectedLayer(Nclasses, 'Name', 'fc2', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10));
net = connectLayers(net, 'avg_pool', 'fc1');
net = connectLayers(net, 'avg_pool', 'fc2');

% and 2 corresponding softmax layers
net = addLayers(net, customSoftmaxLayer('sm1'));
net = connectLayers(net, 'fc1', 'sm1');
net = addLayers(net, customSoftmaxLayer('sm2'));
net = connectLayers(net, 'fc2', 'sm2');

% concatenate softmax output
net = addLayers(net, concatenationLayer(3,2,'Name','concat'));
net = connectLayers(net, 'sm1', 'concat/in1');
net = connectLayers(net, 'sm2', 'concat/in2');

% use 1 cross entropy loss at the end
net = addLayers(net, smoothedCrossEntropyLayer('ce'));
net = connectLayers(net, 'concat', 'ce');


%% Training parameters

miniBatchSize = 32;
L2reg = 0;
lr = 1e-3;  
lrDropRate = 0.1; 
lrDropPeriod = 10;
validationFreq = 300;
maxEpochs = 50;
gradientMomentum = 0.9;
sordDsTrain.MiniBatchSize = miniBatchSize;

    
sordDsTrain.randomCrop = true;
sordDsTrain.horizontalFlip = true;

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
        'ValidationData', sordDsVal, ...
        'ValidationFrequency', validationFreq, ...
        'ValidationPatience', Inf);
    
    if e == 1
    	[net, trainInfo] = trainNetwork(sordDsTrain, net, options);
    else
       [net, trainInfo] = trainNetwork(sordDsTrain, layerGraph(net), options);
    end

    validationLoss(e) = trainInfo.ValidationLoss(end);
    
    if e > 3
        if validationLoss(e) > validationLoss(e-3)
            lr = lr * lrDropRate;
        end
    end
    
    if mod(e, 5) == 0
        name = ['checkpoints/newSord_', date, '_', num2str(e)];
        save(name, 'net', 'trainInfo')
    end
end

%% Make prediction on validation data

sordDsTest.randomCrop = false;
sordDsTest.horizontalFlip = false;
pred = predict(net, sordDsTest);
%% Arg max prediction
% rho = pred(:, 1:Nclasses);
% theta = pred(:, Nclasses+1:end);
% 
% [~, rhomax] = max(rho, [], 2);
% [~, thetamax] = max(theta, [], 2);
% 
% rhoEst = sordDsTest.rhoClasses(rhomax);
% thetaEst = sordDsTest.thetaClasses(thetamax);

%% EV prediction

rho = pred(:, 1:Nclasses);
theta = pred(:, Nclasses+1:end);
rhoEst = sordDsTest.rhoClasses * rho';
thetaEst = sordDsTest.thetaClasses * theta';

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
  
  x1 = width / 2; x2 = width / 2;
  y1Im = -(l(1) * x1 + l(3)) / l(2);
  y2Im = -(l(1) * x2 + l(3)) / l(2);
  
  y1ImHat = -(lhat(1) * x1 + lhat(3)) / lhat(2);
  y2ImHat = -(lhat(1) * x2 + lhat(3)) / lhat(2);
  
  horErr(n) = max(abs([y1ImHat - y1Im, y2ImHat - y2Im])) / height;
  ysave(n, :) = [y1Im, y2Im] / height;
  yHatsave(n, :) = [y1ImHat, y2ImHat] / height;
end
  
%% Calculate empirical cumulative error distribution

do_plot = true;
auc = calc_auc(horErr, do_plot, '', false);

%% Show samples for some images

displayImages = false;
if displayImages
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
        
        Mhat = [x0Hat, y0Hat, 1; x1Hat, y1Hat, 1];
        lhat = null(Mhat);
        lhat(1) = -lhat(1);
        
        x1 = width / 2; x2 = width / 2;
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
end
