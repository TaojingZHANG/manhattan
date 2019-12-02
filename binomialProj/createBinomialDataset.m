%% Create ImageRegressionDataStore from Horizon Lines in the Wild dataset
clear
close all
rng(0);


addpath('../tools/')
horizonDir = '../wildhorizon/';
fileName = 'metadata.csv';

fid = fopen([horizonDir, fileName]);
imdata = textscan(fid, '%s %f %f %f %f %*[^\n]', 'Delimiter', ',');
fid = fclose(fid);


%% Extract split

k = 1; % pick every k:th image

fid = fopen([horizonDir, 'split/train.txt']);
train = textscan(fid, '%s %*[^\n]');
fid = fclose(fid);

train = {(train{1}(1:k:end))}; % use reduced train set for faster training!


fid = fopen([horizonDir, 'split/val.txt']);
val = textscan(fid, '%s %*[^\n]');
fid = fclose(fid);

val = {(val{1}(1:k:end))}; % use reduced validation set for faster training!

fid = fopen([horizonDir, 'split/test.txt']);
test = textscan(fid, '%s %*[^\n]');
fid = fclose(fid);

test = {(test{1}(1:k:end))}; % use reduced test set for faster training!


%% Create normalized horizon labels v = [x, y, z], ||v|| = 1

imSize = 256;
cropSize = 224;
useBw = false;

trainLabels = zeros(4, length(train{1}));
rhoTrain = zeros(1, length(train{1}));
thetaTrain = zeros(1, length(train{1}));

valLabels = zeros(4, length(val{1}));
testLabels = zeros(4, length(test{1}));

for i = 1:length(trainLabels)
  name = train{1}{i};
  index = find(contains(imdata{1}, name));

  x1 = imdata{2}(index);
  y1 = imdata{3}(index);
  x2 = imdata{4}(index);
  y2 = imdata{5}(index);
  trainLabels(:, i) = [imdata{2}(index), imdata{3}(index), imdata{4}(index), imdata{5}(index)];
  
  I = imread([horizonDir, 'images/', name]);
  sz = size(I);
  scale = min(sz(1:2)) / cropSize;
  
  rhoTrain(i) = (x2 * y1 - y2 * x1) / sqrt( (y2-y1)^2 + (x2 - x1)^2) / scale;
  thetaTrain(i) = atand((y2-y1) / (x2 - x1));
  
end


for i = 1:length(valLabels)
  name = val{1}{i};
  index = find(contains(imdata{1}, name));
  valLabels(:, i) = [imdata{2}(index), imdata{3}(index), imdata{4}(index), imdata{5}(index)];
end

for i = 1:length(testLabels)
  name = test{1}{i};
  index = find(contains(imdata{1}, name));
  testLabels(:, i) = [imdata{2}(index), imdata{3}(index), imdata{4}(index), imdata{5}(index)];
end

%% Calculate SORD classes
Nclasses = 100;
q = linspace(0, 1, Nclasses);

rhoClasses = quantile(rhoTrain, q);
thetaClasses = quantile(thetaTrain, q);


%% Create HorionDatastore

imdsTrain = fileDatastore(strcat(horizonDir, 'images/', train{:}), 'ReadFcn', @imread);
imdsVal = fileDatastore(strcat(horizonDir, 'images/', val{:}), 'ReadFcn', @imread);
imdsTest = fileDatastore(strcat(horizonDir, 'images/', test{:}), 'ReadFcn', @imread);

%% Calculate a mean image based on a sample of 100 images

N = 100;
meanImage = zeros(cropSize, cropSize);
inds = randperm(length(imdsTrain.Files), N);

for n = 1:N
  X = imread(imdsTrain.Files{inds(n)});
  
  origSize = size(X);
  if origSize(1) > origSize(2) % vertical
    squareSize = origSize(2);
    L = origSize(1);
    interval = [floor(L / 2) - floor(squareSize / 2):floor(L / 2) + floor(squareSize / 2) - 1];
    Xcrop = X(interval, :, :);
  elseif origSize(2) > origSize(1)
    squareSize = origSize(1);
    L = origSize(2);
    interval = [floor(L / 2) - floor(squareSize / 2):floor(L / 2) + floor(squareSize / 2) - 1];
    Xcrop = X(:, interval, :);
  end
  
  Xsmall = imresize(Xcrop, [cropSize, cropSize]);
  
  if useBw
    Xsmall = rgb2gray(Xsmall);
  end
  
  meanImage = meanImage + double(Xsmall) / N;
end

meanImage = uint8(meanImage);

%% Combine images and labels into datastore object with correct image size
doFlip = true;

binDsTrain = binomialDataStore(imdsTrain, trainLabels, imSize, cropSize, doFlip, rhoClasses, thetaClasses, meanImage, useBw);
binDsVal = binomialDataStore(imdsVal, valLabels, imSize, cropSize, doFlip, rhoClasses, thetaClasses, meanImage, useBw);
binDsTest = binomialDataStore(imdsTest, testLabels, imSize, cropSize, doFlip, rhoClasses, thetaClasses, meanImage, useBw);


%% Save images and labels

save('binDs', 'binDsTrain', 'binDsVal', 'binDsTest', 'k');

