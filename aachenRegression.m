%% Euler Angel (Pitch, Roll) regression on Aachen dataset
clear all
close all
rng(0);

%% Configure cluster
configureAurora;

%% Custom network

inputSize = [267, 400, 1];

layers = [
    imageInputLayer([inputSize], 'Normalization', 'zerocenter')
    
    convolution2dLayer([20, 5], 64 ,'Padding','same')
    reluLayer
    
    convolution2dLayer([20, 5], 64 ,'Padding','same')
    reluLayer
    maxPooling2dLayer([2, 1],'Stride',2)
    
    convolution2dLayer([10, 3], 128, 'Padding','same')
    reluLayer
    maxPooling2dLayer([2, 1],'Stride',2)
    
    convolution2dLayer([5, 2], 128, 'Padding','same')
    reluLayer
    maxPooling2dLayer([2, 2],'Stride',2)
        
    fullyConnectedLayer(2048)
    reluLayer
    
    fullyConnectedLayer(2048)
    reluLayer
    
    fullyConnectedLayer(2)
    xyRegressionLayer('xyRegression')];

%% Generate images and labels

load('aachenDs');


%% Training parameters

miniBatchSize = 16;
aachenDsTrain.MiniBatchSize = miniBatchSize;
validationFreq = floor(length(aachenDsTrain.Labels) / miniBatchSize);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'Plots','none', ...
    'L2Regularization', 0, ...
    'VerboseFrequency', 10, ...
    'ValidationData', aachenDsTest, ...
   'ValidationFrequency', validationFreq, ...
   'ValidationPatience', 20, ...
   'ExecutionEnvironment', 'parallel');
  
  
 %% Train network
[net, trainInfo] = trainNetwork(aachenDsTrain, layers, options);
save(['/lunarc/nobackup/users/axelb/', date, num2str(rand), '.mat'])


%% Prediction

vPred = predict(net, aachenDsTest, 'MiniBatchSize', miniBatchSize, 'ExecutionEnvironment', 'cpu');

%%
testLabels = squeeze(cell2mat(aachenDsTest.Labels));
mu = scaleFacs(:, 1);
sigma = scaleFacs(:, 2);
testAngles = labelsToAngles(testLabels, aachenDsTest.Intrinsics, mu, sigma);
testPredAngles = labelsToAngles(vPred, aachenDsTest.Intrinsics, mu, sigma);
testError = testAngles - testPredAngles;

figure(1), 
subplot(211)
plot(rad2deg(testAngles(:, 1))), hold on, plot(rad2deg(testPredAngles(:, 1)), '--')
legend('True', 'Predicted')
title('Test Error')
hold off
subplot(212)
plot(rad2deg(testAngles(:, 2))), hold on, plot(rad2deg(testPredAngles(:, 2)), '--')
legend('True', 'Predicted')
hold off
%%
figure(2)
subplot(211)
ecdf(rad2deg(abs(testError(:, 1))))
xlabel('Pitch Error Distribution [degrees]')
title('Test Set')
subplot(212)
ecdf(rad2deg(abs(testError(:, 2))))
xlabel('Roll Error Distribution [degrees]')



%% Training prediction
vPredTrain = predict(net, aachenDsTrain, 'ExecutionEnvironment', 'cpu');

%%
mu = scaleFacs(:, 1);
sigma = scaleFacs(:, 2);
trainLabels = squeeze(cell2mat(aachenDsTrain.Labels));
trainAngles = labelsToAngles(trainLabels, aachenDsTrain.Intrinsics, mu, sigma);
trainPredAngles = labelsToAngles(vPredTrain, aachenDsTrain.Intrinsics, mu, sigma);
trainError = trainAngles - trainPredAngles;


figure(3), 
subplot(211)
plot(rad2deg(trainAngles(:, 1))), hold on, plot(rad2deg(trainPredAngles(:, 1)), '--')
legend('True', 'Predicted')
title('Training Error')
hold off
subplot(212)
plot(rad2deg(trainAngles(:, 2))), hold on, plot(rad2deg(trainPredAngles(:, 2)), '--')
legend('True', 'Predicted')
hold off

%%
figure(4)
subplot(211)
ecdf(rad2deg(abs(trainError(:, 1))))
xlabel('Pitch Error Distribution [degrees]')
title('Training Set')
subplot(212)
ecdf(rad2deg(abs(trainError(:, 2))))
xlabel('Roll Error Distribution [degrees]')

%% 

figure(5)
subplot(211)
ecdf(rad2deg(abs(trainAngles(:, 1))))
xlabel('Pitch Data Distribution')
subplot(212)
ecdf(rad2deg(abs(trainAngles(:, 2))))
xlabel('Roll Data Distribution')

%% 
figure(6)
plot(trainLabels(:, 1), trainLabels(:, 2), '.')
xlabel('X Label'), ylabel('Y Label')

%% Calculate r^2 coefficients
ybarTrain = mean(trainAngles);
SStotTrain = sum((trainAngles - ybarTrain).^2);
SSresTrain = sum((trainAngles - trainPredAngles).^2);
r2Train = 1 - SSresTrain ./ SStotTrain;

ybarTest = mean(testAngles);
SStotTest = sum((testAngles - ybarTest).^2);
SSresTest = sum((testAngles - testPredAngles).^2);
r2Test = 1 - SSresTest ./ SStotTest;

%%
figure(7)
subplot(211)
plot(rad2deg(trainAngles(:, 1)), rad2deg(trainPredAngles(:, 1)), '.')
title('Training Set')
xlabel(['Pitch: r^2 = ', num2str(r2Train(1))])
subplot(212)
plot(rad2deg(trainAngles(:, 2)), rad2deg(trainPredAngles(:, 2)), '.')
xlabel(['Roll: r^2 = ', num2str(r2Train(2))])


figure(8)
subplot(211)
plot(rad2deg(testAngles(:, 1)), rad2deg(testPredAngles(:, 1)), '.')
title('Test Set')
xlabel(['Pitch: r^2 = ', num2str(r2Test(1))])
subplot(212)
plot(rad2deg(testAngles(:, 2)), rad2deg(testPredAngles(:, 2)), '.')
xlabel(['Roll: r^2 = ', num2str(r2Test(2))])

%% Visualize first conv layer using deep dream 

layer = 2;
name = net.Layers(layer).Name;

channels = 1:64;
I = deepDreamImage(net,layer,channels,'PyramidLevels',1);

figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'])