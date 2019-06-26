clear
rng(0)

%% Generate N random images with M lines and their intersection as labels

N = 1000;
M = 2;
sigma2 = 0.001;

imres = [78, 78];
lineIms = single(zeros(imres(1), imres(2), 4, N)); % uint8
normLabels = zeros(N, 3);

iCoord = repmat(1:imres(1), [imres(1), 1]);
jCoord = repmat((1:imres(2))', [1, imres(2)]);
rCoord = sqrt((iCoord - imres(1)/2).^2 + (jCoord - imres(2)/2).^2);
% thetaCoord = ?

iCoord = (iCoord - mean(iCoord(:))) / std(iCoord(:));
jCoord = (jCoord - mean(jCoord(:))) / std(jCoord(:));
rCoord = (rCoord - mean(rCoord(:))) / std(rCoord(:));


close all
figure('Position', [100 100 100 100]);
for n = 1:N
  ax = subplot(1,1,1);

  % Create two lines from random points
  a = zeros(M, 1);
  b = zeros(M, 1);
  c = zeros(M, 1);
  for m = 1:2
    theta = 0;
    if m == 1
      p1 = [-1; -1 + 2 * rand()];
      p2 = [1; -1 + 2 * rand()];
    else
      p1 = [-1 + 2 * rand(); -1];
      p2 = [-1 + 2 * rand(); 1];
      % make sure the points are separated enough
    end
    
    % Write line as ax + by + c = 0
    dx = p2(1) - p1(1);
    dy = p2(2) - p1(2);
    
    %theta = atan(dy / dx);
    plot([p1(1), p2(1)], [p1(2), p2(2)], 'w');
    hold on
    
    a(m) = -dy / dx;
    b(m) = 1;
    c(m) = dy / dx * p1(1) - p1(2);
    
  end
  hold off
  
  % Calculate intersection
  A = [a, b];
  r = [-A \ c; 1];
  r = r ./ sqrt(sum(r.^2));
  
  normLabels(n, :) = r(1:3);

  % Save plot
  axis([-1, 1, -1, 1])
  pbaspect([1 1 1])
  set(gca,'visible','off')
  set(gcf,'color','k');
  I = frame2im(getframe(ax));
  Inoisy = imnoise(uint8(rgb2gray(I)), 'gaussian', 0, sigma2);
  lineIms(:, :, 1, n) = Inoisy';
  lineIms(:, :, 2, n) = iCoord;
  lineIms(:, :, 3, n) = jCoord;
  lineIms(:, :, 4, n) = rCoord;
  
end

labels = normLabels';

stdLabels = reshape(labels(1:3, :), [1, 1, 3, N]);
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
