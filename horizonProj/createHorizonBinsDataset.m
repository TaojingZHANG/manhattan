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

trainLabels = zeros(2, length(train{1}));
valLabels = zeros(2, length(val{1}));
testLabels = zeros(2, length(test{1}));

xScale = 5000; %mean(abs(imdata{2}));
yScale = 5000; %mean(abs(imdata{3}));
x1save = zeros(length(trainLabels), 1);
x2save = zeros(length(trainLabels), 1);
y1save = zeros(length(trainLabels), 1);
y2save = zeros(length(trainLabels), 1);

for i = 1:length(trainLabels)
  name = train{1}{i};
  index = find(contains(imdata{1}, name));
  
%   I = imfinfo([horizonDir, 'images/', cell2mat(imdata{1}(index))]);
%   width = I.Width;
%   height = I.Height;
%   
  x1save(i) = imdata{2}(index) / xScale;
  y1save(i) = imdata{3}(index) / yScale;
  x2save(i) = imdata{4}(index) / xScale;
  y2save(i) = imdata{5}(index) / yScale;
  trainLabels(:, i) = [y1save(i); y2save(i)];
end


for i = 1:length(valLabels)
  name = val{1}{i};
  index = find(contains(imdata{1}, name));
  x1 = imdata{2}(index) / xScale;
  y1 = imdata{3}(index) / yScale;
  x2 = imdata{4}(index) / xScale;
  y2 = imdata{5}(index) / yScale;
  valLabels(:, i) = [y1; y2];
end

for i = 1:length(testLabels)
  name = test{1}{i};
  index = find(contains(imdata{1}, name));
  x1 = imdata{2}(index) / xScale;
  y1 = imdata{3}(index) / yScale;
  x2 = imdata{4}(index) / xScale;
  y2 = imdata{5}(index) / yScale;
  testLabels(:, i) = [y1; y2];
end

%% Binning

Nbins = 7;

[trainCats, testCats, centroids] = rectangularBinning(trainLabels, testLabels, Nbins, true);

%% Create HorionDatastore

imdsTrain = fileDatastore(strcat(horizonDir, 'images/', train{:}), 'ReadFcn', @imread);
imdsVal = fileDatastore(strcat(horizonDir, 'images/', val{:}), 'ReadFcn', @imread);
imdsTest = fileDatastore(strcat(horizonDir, 'images/', test{:}), 'ReadFcn', @imread);

%% Calculate a mean image based on a sample of 100 images
imSize = [224, 224];
useBw = false;

N = 100;
meanImage = zeros(imSize);
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
  
  Xsmall = imresize(Xcrop, imSize);
  
  if useBw
    Xsmall = rgb2gray(Xsmall);
  end
  
  meanImage = meanImage + double(Xsmall) / N;
end

meanImage = uint8(meanImage);

%% Combine images and labels into datastore object with correct image size
numClasses = numel(unique(trainCats));

horizonDsTrain = horizonDataStore(imdsTrain, categorical(trainCats'), imSize, meanImage, useBw, numClasses);
horizonDsTest = horizonDataStore(imdsTest, categorical(testCats'), imSize, meanImage, useBw, numClasses);


%% Save images and labels

save('horizonBinsDs', 'horizonDsTrain', 'horizonDsTest', 'k', 'yScale', 'xScale', 'centroids');

