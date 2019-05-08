clear
rng(0)

%% Generate N random images with 1 line and its rotation angle as label

N = 1000;

imres = [342, 342];
lineIms = uint8(zeros(imres(1), imres(2), 1, N));
labels = zeros(N, 1);

a = -1 + 2 * rand(N, 1);
b = -1 + 2 * rand(N, 1);
c = -1 + 2 * rand(N, 1);

close all
figure(1)
for n = 1:N
  ax = subplot(1,1,1);

  % Create 1 line from random points
  p1 = -1 + 2 * rand(2, 1);
  p2 = -1 + 2 * rand(2, 1);
  % make sure the points are separated enough
  while sqrt(sum((p1-p2).^2)) < 0.5
    p2 = -1 + 2 * rand(2, 1);
  end
  plot([p1(1), p2(1)], [p1(2), p2(2)], 'w');
  hold on
  
  % Calculate angle of rotation
  dx = p2(1) - p1(1);
  dy = p2(2) - p1(2);
  
  labels(n) = atan(dy / dx);
    
  hold off

  % Save plot
  axis([-1, 1, -1, 1])
  pbaspect([1 1 1])
  set(gca,'visible','off')
  set(gcf,'color','k');
  I = frame2im(getframe(ax));
  lineIms(:, :, :, n) = uint8(rgb2gray(I));
  
end

%% Normalize labels

mu = mean(labels);
sigma = std(labels);
labels = (labels - mu) / sigma;


%% Reshape

labels = reshape(labels, [1, 1, 1, N]);

%% Split train/test
ratio = 0.9;

train = 1:ceil(ratio * N);
test = ceil(ratio * N):N;

trainIms = lineIms(:, :, :, train);
testIms = lineIms(:, :, :, test);

trainLabels = labels(:, :, :, train);
testLabels = labels(:, :, :, test);

%% Save images and labels
save('lineAnglesData', 'trainIms', 'testIms', 'trainLabels', 'testLabels', ...
  'mu', 'sigma');
