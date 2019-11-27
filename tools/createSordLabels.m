function sordLabels = createSordLabels(labels, classes, phi)

Nclasses = length(classes);
sordLabels = zeros(1, 1, Nclasses, length(labels));

for i = 1:length(labels)
  for n = 1:Nclasses
    sordLabels(1, 1, n, i) = exp(-phi(labels(i), classes(n)));
  end
  if sum(sordLabels(1, 1, :, i)) > 0
    sordLabels(1, 1, :, i) = sordLabels(1, 1, :, i) / sum(sordLabels(1, 1, :, i));
  else
    [~, argmax] = max(-phi(labels(i), classes));
    sordLabels(1, 1, argmax, i) = 1;
  end
end

end

