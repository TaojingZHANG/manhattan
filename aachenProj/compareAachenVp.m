load('aachenDs.mat');

index = 1225;
X = imread(aachenDs.Datastore.Files{index});
vpLabel = squeeze(aachenDs.Labels{index});

f = aachenDs.Intrinsics{4}(index);
cx = aachenDs.Intrinsics{5}(index);
cy = aachenDs.Intrinsics{6}(index);
r = aachenDs.Intrinsics{7}(index);

imSize = size(X);
camParams = cameraIntrinsics(f, [cx, cy], imSize(1:2), ...
  'RadialDistortion', [r, 0]);

Xcal = undistortImage(X, camParams);

% Lines for image index 100
% l1 = [214, 995; 343, 95];
% l2 = [1085, 722; 1094, 890];
% l3 = [770, 303; 765, 460];
% l4 = [895, 303; 898, 483];

% Lines for image index 200
% l1 = [131, 478; 180, 249];
% l2 = [687, 590; 712, 274];
% l3 = [893, 606; 900, 300];
% l4 = [1369, 974; 1350, 737];

% Lines for image index 1225
l1 = [412, 984; 591, 132];
l2 = [748, 853; 744, 457];
l3 = [46, 320; 221, 112];
l4 = [1334, 765; 1203, 486];

l = [l1; l2; l3; l4];

[vhatNorm, vHat] = lsIntersection(l, 1:(length(l) / 2));
plotvlines(Xcal, l, 1:(length(l) / 2), vHat);