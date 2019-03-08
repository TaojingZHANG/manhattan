%% LOAD THE IMAGES FROM THE YORK DATASET
imdir = '.';

clear all
load('YorkUrbanDB/Manhattan_Image_DB_Names');
imageNames = Manhattan_Image_DB_Names;
angleErr = zeros(1, length(imageNames));
angleErrGt = zeros(1, length(imageNames));

plotMode = false;


for p = 1:length(imageNames)
  imageName = Manhattan_Image_DB_Names{p}(1:end-1);
  im = imread(['YorkUrbanDB/', imageName, '/', imageName, '.jpg']);
  load(['YorkUrbanDB/', imageName, '/', imageName, 'LinesAndVP'])
  load(['YorkUrbanDB/', imageName, '/', imageName, 'GroundTruthVP_CamParams'])
  load(['YorkUrbanDB/', imageName, '/', imageName, 'GroundTruthVP_Orthogonal_CamParams'])
  
  %% Find line angles using ground truth lines
  nLines = length(lines) / 2;
  theta = zeros(nLines, 1);
  for k = 1:nLines
    l = lines(2*k-1:2*k, :);
    dx = l(2, 1) - l(1, 1);
    dy = l(2, 2) - l(1, 2);
    theta(k) = atan2(dy, dx);
  end
  
  %% Find lines corresponding to vertical direction
  
  vlines = [find(rad2deg(theta) > 80 & rad2deg(theta) < 100); ...
   find(rad2deg(theta) > -100 & rad2deg(theta) < -80)];
  
  %% Calculate vanishing point using least squares
  vhat = lsIntersection(lines, vlines);
  
  %% Calculate ground truth vanishing point
  gt = find(vp_association == 2);
  vgt = lsIntersection(lines, gt);
  
  %% Calculate angular error
    
  errs = zeros(length(gt), 1);
  errsgt = zeros(length(gt), 1);
  for k = 1:length(gt)
    vk = gt(k);
    l = lines(2*vk-1:2*vk, :);
    M = [l(1, 1), l(1, 2), 1; l(2, 1), l(2, 2), 1];
    m = null(M);
    m = m / norm(m);
    errs(k) = asind(m' * vhat);
    errsgt(k) = asind(m' * vgt);
  end
  angleErr(p) = rms(errs);
  angleErrGt(p) = rms(errsgt);
  
  if plotMode
    close all
    plotvlines(im, lines, vlines, vhat, 'b');
    title('Estimated');
    waitforbuttonpress
    plotvlines(im, lines, gt, vgt, 'b');
    title('Grount Truth');
    waitforbuttonpress
  end
  
end

%%
close all
ecdf(angleErr)
hold on
ecdf(angleErrGt)
legend('My', 'Ground Truth');
xlabel('RMS Angle Error (degrees)')
ylabel('CDF')
