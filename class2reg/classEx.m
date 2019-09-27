clear
close all

[XTrain,~,YTrain] = digitTrain4DArrayData;
[XValidation,~,YValidation] = digitTest4DArrayData;

%%

% classes = (-45:45)';
% trainLabels = zeros(length(YTrain), 1);
% for n = 1:length(trainLabels)
%   trainLabels(n) = find(YTrain(n) == classes);
% end
% 
% valLabels = zeros(length(YValidation), 1);
% for n = 1:length(valLabels)
%   valLabels(n) = find(YValidation(n) == classes);
% end

Nclasses = 91;
trainLabels = categorical(YTrain);
valLabels = categorical(YValidation);


%%

layers = [
  imageInputLayer([28 28 1])
  
  convolution2dLayer(3,8,'Padding','same')
  batchNormalizationLayer
  reluLayer
  
  averagePooling2dLayer(2,'Stride',2)
  
  convolution2dLayer(3,16,'Padding','same')
  batchNormalizationLayer
  reluLayer
  
  averagePooling2dLayer(2,'Stride',2)
  
  convolution2dLayer(3,32,'Padding','same')
  batchNormalizationLayer
  reluLayer
  
  convolution2dLayer(3,32,'Padding','same')
  batchNormalizationLayer
  reluLayer
  
  dropoutLayer(0.2)
  fullyConnectedLayer(Nclasses)
  softmaxLayer
  classificationLayer];

miniBatchSize  = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm',...
  'MiniBatchSize',miniBatchSize,...
  'MaxEpochs',30,...
  'InitialLearnRate',1e-3,...
  'LearnRateSchedule','piecewise',...
  'LearnRateDropFactor',0.1,...
  'LearnRateDropPeriod',20,...
  'Shuffle','every-epoch',...
  'ValidationData',{XValidation,valLabels},...
  'ValidationFrequency',validationFrequency,...
  'ValidationPatience',Inf,...
  'Plots','training-progress',...
  'Verbose',false);

net = trainNetwork(XTrain,trainLabels,layers,options);

%%

YPredicted = classify(net,XValidation);

%accuracy = sum(YPredicted == valLabels)/numel(valLabels)

predictionError = double(valLabels) - double(YPredicted);

thr = 10;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(valLabels);

accuracy = numCorrect/numValidationImages

squares = predictionError.^2;
rmse = sqrt(mean(squares))
