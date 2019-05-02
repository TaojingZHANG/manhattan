function plotvlines(im, lines, indices, vpoints, plotColor)

if nargin < 5
  plotcolor = 'b';
end

%% Show image
figure
imshow(im)
hold on

%% Plot lines
for k = 1:length(indices)
  vk = indices(k);
  l = lines(2*vk-1:2*vk, :);
  plot(l(:, 1), l(:, 2), 'b', 'LineWidth', 5)
  hold on
end

%% Find lines as p = a + tn
figure
hold on

a = zeros(2, length(indices));
n = zeros(2, length(indices));
for k = 1:length(indices)
  vk = indices(k);
  l = lines(2 * vk-1:2 * vk, :);
  a(:, k) = l(1, :);
  dx = l(2, 1) - l(1, 1);
  dy = l(2, 2) - l(1, 2);
  n(:, k) = [dx; dy] / sqrt(dx^2 + dy^2);
end

Npoints = 100;
t = linspace(-10000, 1000, Npoints);
t = [t; t];
for k = 1:size(a, 2)
  line = repmat(a(:, k), [1, Npoints]) + t.* repmat(n(:, k), [1, Npoints]);
  plot(line(1, :), line(2, :), plotColor)
end

%% Plot vanishing point
for i = 1:size(vpoints, 2)
  plot(vpoints(1, i) / vpoints(3, i), vpoints(2, i) / vpoints(3, i), ...
    '*', 'MarkerSize', 10)
end

end

