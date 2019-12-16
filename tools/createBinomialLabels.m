function binLabels = createBinomialLabels(labels, classes)

Nclasses = length(classes);
L = floor(Nclasses / 4);

binLabels = zeros(Nclasses, length(labels));
for i = 1:length(labels)
  d = abs(labels(i) - classes);
  [~, argmin] = min(d);
  binLabels(mod(argmin - 1 + (-L:L), Nclasses) + 1, i) = 1;
end

binLabels = reshape(binLabels, [1, 1, Nclasses, length(labels)]);

end

