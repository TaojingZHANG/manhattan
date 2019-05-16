clear
rng(0)

%% Generate N random images with M lines and their intersection as labels

N = 1000;
M = 2;
sigma2 = 0.01;

imres = [78, 78];
lineIms = uint8(zeros(imres(1), imres(2), 1, N));
normLabels = zeros(N, 2);

close all
figure('Position', [100 100 100 100]);
for n = 1:N
  ax = subplot(1,1,1);

  % Create two lines from random points
  a = zeros(M, 1);
  b = zeros(M, 1);
  c = zeros(M, 1);
  for m = 1:M
    theta = 0;
    while abs(theta) < pi/4
      p1 = -1 + 2 * rand(2, 1);
      p2 = -1 + 2 * rand(2, 1);
      % make sure the points are separated enough
      while sqrt(sum((p1-p2).^2)) < 0.5
        p2 = -1 + 2 * rand(2, 1);
      end
      
      % Write line as ax + by + c = 0
      dx = p2(1) - p1(1);
      dy = p2(2) - p1(2);
      
      theta = atan(dy / dx);
    end
    plot([p1(1), p2(1)], [p1(2), p2(2)], 'w');
    hold on
    
    a(m) = -dy / dx;
    b(m) = 1;
    c(m) = dy / dx * p1(1) - p1(2);
    
  end
  hold off

  % Save plot
  axis([-1, 1, -1, 1])
  pbaspect([1 1 1])
  set(gca,'visible','off')
  set(gcf,'color','k');
  I = frame2im(getframe(ax));
  lineIms(:, :, :, n) = imnoise(uint8(rgb2gray(I)), 'gaussian', 0, sigma2);
      
  % Calculate intersection
  A = [a, b];
  r = [-A \ c; 1];
  rnorm = r / norm(r);
  pitch = atan(-r(1) / sqrt(r(2)^2 + r(3)^2));
  roll = atan(r(2) / r(3));
  normLabels(n, :) = [pitch; roll];
  
end

% X-axis: 2 units = 560 pixels --> 1 pixel = 
% Y-axis 2 units = 420 pixels
% px = 2 / imres(1);
% py = 2 / imres(2);
% 
% % Principal point
% cx = imres(1) / 2;
% cy = imres(2) / 2;
% 
% K = [1/ px, 0, cx; 0, -1 / py, cy; 0, 0, 1];
labels = normLabels'; % normLaK * normLabels';

% % %% Remove outliers
% mu = mean(labels(1:2, :), 2);
% sigma = std(labels(1:2, :), [], 2);
% 
% outliers = [find(abs(labels(1, :)) > mu(1) + 2 * sigma(1)), ...
%         find(abs(labels(2, :)) > mu(2) + 2 * sigma(2))];
%       
% labels(:, outliers) = [];
% lineIms(:, :, :, outliers) = [];
% 
% %% Normalize labels
% 
% muScale = mean(labels(1:2, :), 2);
% sigmaScale = std(labels(1:2, :), [], 2);
% 
% stdLabels = (labels(1:2, :) - muScale) ./ sigmaScale;
% 
% %% Remove outliers again
% mu = mean(stdLabels(1:2, :), 2);
% sigma = std(stdLabels(1:2, :), [], 2);
% 
% outliers = [find(abs(stdLabels(1, :)) > mu(1) + 2 * sigma(1)), ...
%         find(abs(stdLabels(2, :)) > mu(2) + 2 * sigma(2))];
%       
% stdLabels(:, outliers) = [];
% lineIms(:, :, :, outliers) = [];
% 
% 
% %% Reshape
% 
% Nnew = length(stdLabels);
% stdLabels = reshape(stdLabels, [1, 1, 2, Nnew]);

stdLabels = reshape(labels(1:2, :), [1, 1, 2, N]);
muScale = 1;
sigmaScale = 1;
Nnew = N;

%% Split train/test
ratio = 0.9;

train = 1:ceil(ratio * Nnew);
test = ceil(ratio * Nnew):Nnew;

trainIms = lineIms(:, :, :, train);
testIms = lineIms(:, :, :, test);

trainLabels = stdLabels(:, :, :, train);
testLabels = stdLabels(:, :, :, test);

%% Save images and labels
save('lineData', 'trainIms', 'testIms', 'trainLabels', 'testLabels', ...
  'muScale', 'sigmaScale');
