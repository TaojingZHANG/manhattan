%% Extract rotation labels from Aachen dataset

aachenDir = 'aachen/';
fileName = 'aachen_cvpr2018_db.nvm';

fid = fopen([aachenDir, fileName]);
labels = textscan(fid, '%s %*f %f %f %f %f %*[^\n]','HeaderLines',3);
fid = fclose(fid);
interval = 1:4328;
names = labels{1}(interval);

%% Extract intrisics
fid = fopen([aachenDir, 'database_intrinsics.txt']);
intrinsics = textscan(fid, '%s %*s %f %f %f %f %f %f %*[^\n]','HeaderLines',0);
fid = fclose(fid);


%% Extract rotation quaternions
quats = zeros(4, length(interval));
for i = interval
  quats(1, i) = labels{2}(i);
  quats(2, i) = labels{3}(i);
  quats(3, i) = labels{4}(i);
  quats(4, i) = labels{5}(i);
end

%% Calculate Euler angles: pitch, roll and yaw
eulerAngles = zeros(3, length(quats));
newQuats = zeros(4, length(quats));
rots = zeros(3, length(quats));
for i = 1:length(quats)
  q0 = quats(1, i);
  q1 = quats(2, i);
  q2 = quats(3, i);
  q3 = quats(4, i);
  q = quaternion(q0, q1, q2, q3);
  eulerAngles(:, i) = euler(q, 'YZX', 'frame');
  
  qnew = quaternion([0, eulerAngles(2:3, i)'], 'euler', 'YZX', 'frame');
  qnew = compact(qnew);
  newQuats(1, i) = qnew(1);
  newQuats(2, i) = qnew(2);
  newQuats(3, i) = qnew(3);
  newQuats(4, i) = qnew(4);
  
  if eulerAngles(3, i) > 0
    eulerAngles(3, i) = eulerAngles(3, i) - pi;
  else
    eulerAngles(3, i) = eulerAngles(3, i) + pi;
  end
%   eulerAngles(1, i) = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1^2 + q2^2));
%   eulerAngles(2, i) = asin(2 * (q0 * q2 - q3 * q1));
%   eulerAngles(3, i) = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2^2 + q3^2));

  R = quat2rotm(q);
  f = intrinsics{4}(i);
  cx = intrinsics{5}(i);
  cy = intrinsics{6}(i);
  
  K = [f, 0, cx; 0, f, cy; 0, 0, 1];
  
  KR = K * R;
  rots(:, i) = KR(:, 2) ; % second column is projections of vanishing point in y-direction
  
end

%% Create image datastore object

imdsTrain = fileDatastore([aachenDir, 'images_upright/db'], 'ReadFcn', @imread, ...
  'FileExtensions', '.jpg');


labels = zeros(1, 1, 2, length(imdsTrain.Files));
camParams = cell(7, 1);
camParams{1} = cell(length(imdsTrain.Files), 1);
for j = 2:length(camParams)
  camParams{j} = zeros(length(imdsTrain.Files), 1);
end
for i = 1:length(quats)
  n = names{i};
  index = find(contains(imdsTrain.Files, n));
%   labels(1, 1, 1, index) = eulerAngles(2, i);
%   labels(1, 1, 2, index) = eulerAngles(3, i);
  w = intrinsics{2}(i);
  h = intrinsics{3}(i);
  if h == 1067 && w == 1600 % only use horizontal images
    labels(1, 1, :, index) = rots(1:2, i) ./ rots(3, i);
    for j = 1:length(camParams)
      camParams{j}(index) = intrinsics{j}(i);
    end
  end
end


emptyLabels = find(labels(1, 1, 1, :) == 0);
imdsTrain.Files(emptyLabels) = [];
labels(:, :, :, emptyLabels) = [];
for j = 1:length(camParams)
  camParams{j}(emptyLabels) = [];
end

scaleFacs = [mean(labels(:, :, 1, :)),  std(labels(:, :, 1, :)); ...
     mean(labels(:, :, 2, :)),  std(labels(:, :, 2, :))];

normLabels = zeros(size(labels));
normLabels(:, :, 1, :) = (labels(:, :, 1, :) - mean(labels(:, :, 1, :))) / std(labels(:, :, 1, :));
normLabels(:, :, 2, :) = (labels(:, :, 2, :) - mean(labels(:, :, 2, :))) / std(labels(:, :, 2, :));

for n = 1:2
    mux = median(normLabels(:, :, 1, :));
    muy = median(normLabels(:, :, 2, :));
    
    outliers = [find(abs(normLabels(1, 1, 1, :)) > mux + 2 * std(normLabels(1, 1, 1, :))); ...
        find(abs(normLabels(1, 1, 2, :)) > muy + 2 * std(normLabels(1, 1, 2, :)))];
    
    imdsTrain.Files(outliers) = [];
    normLabels(:, :, :, outliers) = [];
    
    for i = 1:length(intrinsics)
        camParams{i}(outliers) = [];
    end
end


%% Split training and validation
imdsTest = copy(imdsTrain);
trainLabels = normLabels;
testLabels = normLabels;
shuffleInds = randperm(length(normLabels));%length(normLabels)*0.9;
train = shuffleInds(1:floor(length(normLabels)*0.9));
test = shuffleInds(length(train) + 1:length(normLabels));

imdsTrain.Files(test) = [];
trainLabels(:, :, :, test) = [];
camParamsTrain = camParams;
for i = 1:length(intrinsics)
  camParamsTrain{i}(test) = [];
end

imdsTest.Files(train) = [];
testLabels(:, :, :, train) = [];
camParamsTest= camParams;
for i = 1:length(intrinsics)
  camParamsTest{i}(train) = [];
end


%% Combine images and labels
aachenDsTrain = imageRegressionDatastore(imdsTrain.Files, trainLabels, camParamsTrain);
aachenDsTest = imageRegressionDatastore(imdsTest.Files, testLabels, camParamsTest);


%% Save images and labels

save('aachenDs', 'aachenDsTrain', 'aachenDsTest', 'scaleFacs');
