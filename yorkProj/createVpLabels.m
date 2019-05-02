%% Takes the ground truth vertical vps and stores them as labels
%% YorkUrbanDB
imdir = '.';

clear all
load('YorkUrbanDB/Manhattan_Image_DB_Names');
imageNames = Manhattan_Image_DB_Names;
vpLabels = zeros(3, length(imageNames));
vpOrthLabels = zeros(2, length(imageNames));
vgtLabels = zeros(2, length(imageNames));
prLabels = zeros(2, length(imageNames));
imData = zeros(480, 640, 1, length(imageNames));


for p = 1:length(imageNames)
  imageName = Manhattan_Image_DB_Names{p}(1:end-1);
  im = imread(['YorkUrbanDB/', imageName, '/', imageName, '.jpg']);
  load(['YorkUrbanDB/', imageName, '/', imageName, 'GroundTruthVP_CamParams'])
  load(['YorkUrbanDB/', imageName, '/', imageName, 'GroundTruthVP_Orthogonal_CamParams'])
  load(['YorkUrbanDB/', imageName, '/', imageName, 'LinesAndVP'])
  vpLabel = vp(1:3, 2);
  vpLabels(:, p) = vpLabel;
  vpOrthLabel = vp_orthogonal(1:2, 2);
  vpOrthLabels(:, p) = vpOrthLabel;
  gt = find(vp_association == 2);
  
  [~, vgt] = lsIntersection(lines, gt);
  
  vgtLabels(:, p) = vgt(1:2);
  
  pitch = atan(-vp(1, 2) / sqrt(vp(2, 2)^2 + vp(3, 2)^2));
  roll = mod(atan(vp(2, 2) / vp(3, 2)), pi) - pi / 2; % (90 degrees?)
  %roll = atan2(vp(3, 2), vp(2, 2));
  
  prLabels(:, p) = [pitch; roll];
  
  imData(:, :, :, p) = rgb2gray(im);
   
end

%vpLabels = (vpLabels - mean(vpLabels, 2)) ./ std(vpLabels, [], 2);
%vpOrthLabels = (vpOrthLabels - mean(vpOrthLabels, 2)) ./ std(vpOrthLabels, [], 2);

save('YorkVpLabels', 'vpLabels', 'vpOrthLabels', 'vgtLabels', 'prLabels', 'imData');


%% PKUCampusDB
imdir = '.';

clear all
load('PKUCampusDB/PKUCampus_Image_DB_Names.mat');
imageNames = PKUCampus_Image_DB_Names;
vpLabels = cell(1, length(imageNames));
vpOrthLabels = zeros(2, length(imageNames));
vgtLabels = zeros(2, length(imageNames));


for p = 1:length(imageNames)
  imageName = PKUCampus_Image_DB_Names{p}(1:end-1);
  %im = imread(['PKUCampusDB/', imageName, '/', imageName, '.jpg']);
  load(['PKUCampusDB/', imageName, '/', imageName, 'GroundTruthVP_CamParams'])
  load(['PKUCampusDB/', imageName, '/', imageName, 'GroundTruthVP_Orthogonal_CamParams'])
  load(['PKUCampusDB/', imageName, '/', imageName, 'LinesAndVP'])
  vpLabel = v(1:2, 2);
  vpLabels{p} = vpLabel / norm(vpLabel);
  vpOrthLabel = vp_orthogonal(1:2, 2);
  vpOrthLabels(:, p) = vpOrthLabel / norm(vpOrthLabel);
  gt = find(vp_association == 2);
  vgt = lsIntersection(lines, gt);
  vgtLabels(:, p) = vgt(1:2) / norm(vgt(1:2));
  %plotvlines(im, lines, gt, vgt);

  
end


save('PKUVpLabels', 'vpLabels', 'vpOrthLabels', 'vgtLabels');

