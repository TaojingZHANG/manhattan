function [trainIms, testIms, trainLabels, testLabels] = notsomeanGen(imSize, sigma)
% Generate images from the Not-so-Clevr dataset

N = imSize^2 * (imSize^2 - 1);

images = zeros(imSize, imSize, 1, N);
labels = zeros(1, 1, 2, N);

space = linspace(-1, 1, imSize);

c = 1;
for i = 1:imSize
  for j = 1:imSize
    for iprime = 1:imSize
      for jprime = 1:imSize
        if ~(iprime == i && jprime == j && j == jprime && i == iprime)
          images(i, j, 1, c) = 1;          
          images(iprime, jprime, 1, c) = -1;
          labels(1, 1, :, c) = 1/2*(space([i, j])) + 1/2*(space([iprime, jprime]));
          images(:, :, 1, c) = images(:, :, 1, c) + sigma * randn(imSize);
          
          c = c + 1;
        end
      end
    end
  end
end

N = c - 1;

split = 'uniform';

if strcmp(split, 'uniform')
  ratio = 0.8;
  shuffled = randperm(N);
  train = shuffled(1:ceil(ratio * N));
  test = shuffled(ceil(ratio * N)+1:N);
  trainIms = images(:, :, :, train);
  testIms = images(:, :, :, test);
  trainLabels = labels(:, :, :, train);
  testLabels = labels(:, :, :, test);
else
  error('invalid split')
end

end