clear
close all
rng(0)

[XTrain,~,YTrain] = digitTrain4DArrayData;
[XValidation,~,YValidation] = digitTest4DArrayData;

%w = -45:0.1:45;
w = [-45, 45];
Nclasses = length(w);

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
  tempLayer('temp', 10);
  softmaxLayer
  class2RegLayer('class2reg', w);
  regressionLayer];

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
  'ValidationData',{XValidation,YValidation},...
  'ValidationFrequency',validationFrequency,...
  'ValidationPatience',Inf,...
  'Plots','training-progress',...
  'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);

%%

YPredicted = predict(net,XValidation);

predictionError = YValidation - YPredicted;

thr = 10;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(YValidation);

accuracy = numCorrect/numValidationImages

squares = predictionError.^2;
rmse = sqrt(mean(squares))
