imdir = '.';

clear all
load('YorkUrbanDB/Manhattan_Image_DB_Names');

imNmbr = 30;
imageName = Manhattan_Image_DB_Names{imNmbr}(1:end-1);
im = imread(['YorkUrbanDB/', imageName, '/', imageName, '.jpg']);
load(['YorkUrbanDB/', imageName, '/', imageName, 'LinesAndVP'])
load(['YorkUrbanDB/', imageName, '/', imageName, 'GroundTruthVP_CamParams'])
load(['YorkUrbanDB/', imageName, '/', imageName, 'GroundTruthVP_Orthogonal_CamParams'])

close all
imshow(im)

%% Calculate ground truth vanishing point
gt = find(vp_association == 2);
vgt = lsIntersection(lines, gt);
vgt = vp_orthogonal(:, 2);

%% Normalize with inv(K)
load('YorkUrbanDB/cameraParameters.mat');

f = focal / pixelSize;
K = [f, 0, pp(1,1); 0, f, pp(1,2); 0, 0, 1];

%% Construct R

yaw = 0;
vgt = K \ vgt;
vgt = vgt / norm(vgt);
pitch = atan(-vgt(1) / sqrt(vgt(2)^2 + vgt(3)^2));
roll = atan(vgt(2) / vgt(3)); % (90 degrees?)

Ryaw = [cos(yaw), -sin(yaw), 0; sin(yaw), cos(yaw), 0; 0, 0, 1]; %z-axis
Rpitch = [cos(pitch), 0, sin(pitch); 0, 1, 0; -sin(pitch), 0, cos(pitch)]; %y-axis
Rroll = [1, 0, 0; 0, cos(roll), -sin(roll); 0, sin(roll), cos(roll)]; %x-axis

R = (Ryaw * Rpitch * Rroll);

%% Construct H using arbtrary hight

d = 1;
t = - R * [0; 0; d];
P = [R, t];

H = [P(:, 1:2), P(:, 4)];

%% Transform using homography
load(['BW', num2str(imNmbr)]);
[x, y] = find(BW);
xbar = K \ [y, x, ones(length(x), 1)].'; %third coordinate is 1, K does not change this
X = H' * xbar;
X = X ./ X(end, :);

close all
figure
imshow(im)
figure
imshow(BW)
figure
plot(xbar(1, :), xbar(2, :), '.')
figure
plot(-X(1, :),  X(2, :), '.')

Pcell = cell(1,1);
Pcell{1} = P;
figure
plotcams(Pcell);

