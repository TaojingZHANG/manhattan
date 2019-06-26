function [trainIms, testIms, trainLabels, testLabels] = notsoclevrGen(split, imSize, squareSize)
% Generate images from the Not-so-Clevr dataset

if nargin < 2
  imSize = 64;
end
if nargin < 3
  squareSize = 9;
end

space = linspace(-1, 1, imSize);
s = squareSize - 1;

imCenters = imSize - s + 1;
N = imCenters^2;

images = zeros(imSize, imSize, 1, N);
labels = zeros(1, 1, 2, N);

c = 1;
for i = 1:imCenters
  for j = 1:imCenters
    images(i:i+s, j:j+s, 1, c) = 1;
    labels(1, 1, :, c) = space([i + s/2, j + s/2]);
    c = c + 1;
  end
end

N = length(images);

if strcmp(split, 'uniform')
  ratio = 0.8;
  shuffled = randperm(N);
  train = shuffled(1:ceil(ratio * N));
  test = shuffled(ceil(ratio * N)+1:N);
  trainIms = images(:, :, :, train);
  testIms = images(:, :, :, test);
  trainLabels = labels(:, :, :, train);
  testLabels = labels(:, :, :, test);
elseif strcmp(split, 'quadrant')
  fourthQuadrant = find(labels(1, 1, 1, :) > imSize / 2 & ...
    labels(1, 1, 2, :) > imSize / 2);
  fourthQuadrant = fourthQuadrant(randperm(length(fourthQuadrant)));
  testLabels = labels(:, :, :, fourthQuadrant);
  testIms = images(:, :, :, fourthQuadrant);
  
  theRest = find(labels(1, 1, 1, :) <= imSize / 2 | ...
    labels(1, 1, 2, :) <= imSize / 2);
  theRest = theRest(randperm(length(theRest)));
  trainLabels = labels(:, :, :, theRest);
  trainIms = images(:, :, :, theRest);
  
else
  error('invalid split')
end


end