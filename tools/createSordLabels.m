function sordLabels = createSordLabels(labels, classes, phi, useModulo360)

if nargin < 4
    useModulo360 = false;
end

Nclasses = length(classes);
sordLabels = zeros(1, 1, Nclasses, length(labels));

for i = 1:length(labels)
    for n = 1:Nclasses
        if useModulo360
            d1 = phi(labels(i), classes(n));
            d2 = phi(mod(labels(i) - classes(n) - 180, 360), 0);
            d = min(d1,d2);
            sordLabels(1, 1, n, i) = exp(-d);
        else
            sordLabels(1, 1, n, i) = exp(-phi(labels(i), classes(n)));
        end
    end
  if sum(sordLabels(1, 1, :, i)) > 0
    sordLabels(1, 1, :, i) = sordLabels(1, 1, :, i) / sum(sordLabels(1, 1, :, i));
  else
    [~, argmax] = max(-phi(labels(i), classes));
    sordLabels(1, 1, argmax, i) = 1;
  end
end

end

