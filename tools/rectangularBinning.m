function [trainCats, testCats, centroids] = rectangularBinning(trainLabels, testLabels, Nbins, doPlot)

if nargin < 4
  doPlot = false;
end

bins = linspace(0, 1, Nbins + 1);
[F1, X1] = ecdf(trainLabels(1, :)');
[F2, X2] = ecdf(trainLabels(2, :)');

X = zeros(size(trainLabels'));
Xtest = zeros(size(testLabels'));
for n = 1:length(X)
  x1 = find(trainLabels(1, n) == X1);
  x2 = find(trainLabels(2, n) == X2);
  X(n, :) = [F1(x1(1)); F2(x2(1))];
end
  
for n = 1:length(Xtest)
  [~, x1test] = min(abs(testLabels(1, n) - X1));
  [~, x2test] = min(abs(testLabels(2, n) - X2));
  Xtest(n, :) = [F1(x1test(1)); F2(x2test(1))];
end

Xnew = zeros(size(X));
Xnewtest = zeros(size(Xtest));

for m = 2:length(bins)
  currInds = X(:, 1) < bins(m) & X(:, 1) >= bins(m - 1);
  x = X(currInds, 2);
  [F1new, X1new] = ecdf(x); 
  
  currTestInds = Xtest(:, 1) < bins(m) & Xtest(:, 1) >= bins(m - 1);
  xtest = Xtest(currTestInds, 2);
  
  xnew = zeros(size(x));
  xnewtest = [];
  for n = 1:length(x)
    x1 = find(x(n) == X1new);
    xnew(n) = F1new(x1(1));
  end
    
  for n = 1:length(xtest)
    [~, x1test] = min(abs(xtest(n) - X1new));
    xnewtest = [xnewtest; F1new(x1test(1))];
  end
  Xnew(currInds, 2) = xnew;
  Xnew(currInds, 1) = X(currInds, 1);
  
  Xnewtest(currTestInds, 2) = xnewtest;
  Xnewtest(currTestInds, 1) = Xtest(currTestInds, 1);
end

trainCats = zeros(length(trainLabels), 1);
for n = 1:length(Xnew)
  for i = 2:length(bins)
    for j = 2:length(bins)
      if Xnew(n, 1) < bins(i) && Xnew(n, 1) >= bins(i - 1) && ...
          Xnew(n, 2) < bins(j) && Xnew(n, 2) >= bins(j - 1)
        trainCats(n) = sub2ind([Nbins, Nbins], i-1, j-1);
      end
    end
  end
end

testCats = zeros(length(testLabels), 1);
for n = 1:length(Xnewtest)
  for i = 2:length(bins)
    for j = 2:length(bins)
      if Xnewtest(n, 1) < bins(i) && Xnewtest(n, 1) >= bins(i - 1) && ...
          Xnewtest(n, 2) < bins(j) && Xnewtest(n, 2) >= bins(j - 1)
        testCats(n) = sub2ind([Nbins, Nbins], i-1, j-1);
      end
    end
  end
end

l = 0:Nbins^2;
centroids = zeros(length(l), 2);
for n = 1:length(l)
  currX = trainLabels(:, trainCats == l(n))';
  centroids(n, :) = mean(currX);
end


if doPlot
  
  histogram2(X(:, 1), X(:, 2), bins, bins);
  
  figure
  histogram2(Xnew(:, 1), Xnew(:, 2), bins, bins);
  
  c = jet(Nbins^2);
  crand = randperm(Nbins^2);
  l = 1:Nbins^2;
  figure
  for n = 1:length(l)
    currX = trainLabels(:, trainCats == l(n))';
    plot(currX(:, 1), currX(:, 2), '.', 'color', c(crand(n), :))
    hold on
  end
  title('Train');
  
  figure
  for n = 1:length(l)
    currX = testLabels(:, testCats == l(n))';
    plot(currX(:, 1), currX(:, 2), '.', 'color', c(crand(n), :))
    hold on
  end
  title('Test')
end

end

