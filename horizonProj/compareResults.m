close all
clear
load('alexnet31augustNorm.mat')
stats = load('stats');

%%

err = zeros(length(pred), 1);
for n = 1:length(pred)
  label = squeeze(horizonDsTest.Labels{n})';
  err(n, :) = sum(cross(label, pred(n, :)).^2);
end

ecdf(err)

err2 = zeros(length(stats.pred), 1);
for n = 1:length(stats.pred)
  label = squeeze(stats.horizonDsTest.Labels{n})';
  err2(n, :) = sum(cross(label, stats.pred(n, :)).^2);
end

figure(1)
hold on
ecdf(err2)
grid on
xlabel('Loss Value (sin^2(\theta))')
ylabel('Cumulative Frequency')
legend('Small dataset: mean test loss = 0.434', 'Large dataset: mean test loss = 0.340')

%%
figure(2)
auc1 = calc_auc(horErr, true, '', false);
hold on
auc2 = calc_auc(stats.horErr, true, '', false);
legend('Small dataset: AUC = 51.3', 'Large dataset: AUC = 51.7')
xlabel('Normalized horizon error score')
ylabel('Cumulative frequency')



%% 
figure(3)
t1 = lowpass(trainInfo.TrainingLoss,0.01);
v1 = trainInfo.ValidationLoss;
v1 = v1(~isnan(v1));
plot(t1)
hold on
plot(1:100:100*length(v1), v1, '-*')

t2 = lowpass(stats.trainInfo.TrainingLoss(1:5:end),0.01);
v2 = stats.trainInfo.ValidationLoss;
v2 = v2(~isnan(v2));
v2 = v2(1:5:end);
plot(t2, '-.')
plot(1:100:100*length(v2), v2, '-+')

xlabel('Iteration')
ylabel('Loss Value (sin^2(\theta))')

legend('Small dataset, training loss', 'Small dataset, validation loss', ...
  'Large dataset, training loss', 'Large dataset, validation loss')
grid on
