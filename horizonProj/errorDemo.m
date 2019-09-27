l = [0; 1; 0];

if l(2) == 0
  error('b should not be 0');
end

l = l / norm(l);

N = 200;
x1 = -1;
x2 = 1;
lHat = zeros(N, 3);
d = zeros(N, N);
loss = zeros(N, N);
[X, Y, Z] = sphere(N);
for n = 1:N
  for nprime = 1:N
%   v = [0; 0; 0];
% 
%     while norm(v) < .0001
%         x = randn();
%         y = randn();
%         z = randn();
%         v = [x, y, z];
%     end
    
    %lhat = v / norm(v);
    lhat = [X(n, nprime); Y(n, nprime); Z(n, nprime)];
    
    d1 = abs((lhat(1) * x1 + lhat(3)) / lhat(2) - (l(1) * x1 + l(3)) / l(2));
    d2 = abs((lhat(1) * x2 + lhat(3)) / lhat(2) - (l(1) * x2 + l(3)) / l(2));
    
    d(n, nprime) = max([d1, d2]);
    loss(n, nprime) = asin(sqrt(sum((cross(l, lhat)).^2)));
    lHat(n, :) = lhat;
  end
end


close all
d(d > pi/2) = pi/2;
surf(X, Y, Z, (d));
xlabel('x'), ylabel('y'), zlabel('z')
title('Normalized Horizon Error Metric')
colorbar
view(l)

figure
surf(X, Y, Z, loss);
xlabel('x'), ylabel('y'), zlabel('z')
title('\theta Metric')
colorbar
view(l)

figure
surf(X, Y, Z, loss - d)
xlabel('x'), ylabel('y'), zlabel('z')
title('Difference Metric')
colorbar
view(l)

%%

% d1 = loss(:);
% d2 = d(:);
% list = [];
% for n = 1:10000
%   for m = n:10000
%     if d1(n) > d1(m) && d2(n) < d2(m)
%       list = [list; n, m];
%     end
%   end
% end

theta1 = pi/2; theta2 = pi / 4;

    
r = (linspace(0.1, 10, 100));

x1 = cos(theta1); x2 = cos(theta2);
z1 = sin(theta1); z2 = sin(theta2);

l1List = [r * x1; ones(1, length(r)); r * z1]; l1List = l1List ./ sqrt(sum(l1List.^2, 1));
l2List = [r * x2; ones(1, length(r)); r * z2]; l2List = l2List ./ sqrt(sum(l2List.^2, 1));

l = [0; 1; 0];
l = l / norm(l);
errHor = zeros(length(r), 2);
errDist = zeros(length(r), 2);
for n = 1:length(r)
  
  l1 = l1List(:, n); l2 = l2List(:, n);
  x1 = 1; x2 = -1;
  errHor(n, 1) = max([abs((l1(1) * x1 + l1(3)) / l1(2) - (l(1) * x1 + l(3)) / l(2)), ...
    abs((l1(1) * x2 + l1(3)) / l1(2) - (l(1) * x2 + l(3)) / l(2))]);
   errHor(n, 2) = max([abs((l2(1) * x1 + l2(3)) / l2(2) - (l(1) * x1 + l(3)) / l(2)), ...
    abs((l2(1) * x2 + l2(3)) / l2(2) - (l(1) * x2 + l(3)) / l(2))]);
  
  errDist(n, 1) = asin(sqrt(sum((cross(l, l1)).^2)));
  errDist(n, 2) = asin(sqrt(sum((cross(l, l2)).^2)));

end


list = [];
for n = 1:length(r)
  for m = 1:length(r)
    if errDist(n, 2) < errDist(m, 1) && errHor(n, 2) > errHor(m, 1)
      list = [list; n, m, errDist(n, 2) - errDist(m, 1), errHor(n, 2) - errHor(m, 1)];
    end
  end
end