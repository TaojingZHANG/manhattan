function vhat = lsIntersection(lines, indices)

 %% Find lines as p = a + tn
  
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
  
  %% Find least squares intersection
  
  K = size(n, 2);
  R = zeros(2, 2);
  q = zeros(2, 1);
  for j = 1:K
    R = R + eye(2) - n(:, j) * n(:, j)';
    q = q + (eye(2) - n(:, j) * n(:, j)') * a(:, j);
  end
  
  phat = R \ q;
  vhat = [phat; 1];
  vhat = vhat / norm(vhat);

end

