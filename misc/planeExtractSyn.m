clear

%% Define plane in Z = 0
Xx = linspace(3, 4, 100);
Xy = linspace(-2, -3, 100);
[Xtemp, Ytemp] = meshgrid(Xx, Xy);%; zeros(1, length(Xx)), ones(1, length(Xx))];
X = [Xtemp(:).'; Ytemp(:).'; zeros(1, numel(Xtemp)); ones(1, numel(Xtemp))];

%% Create camera

trueYaw = 0;
truePitch = -70;
trueRoll = 10;
yaw = deg2rad(trueYaw);
pitch = deg2rad(truePitch);
roll = deg2rad(trueRoll);
Ryaw = [cos(yaw), -sin(yaw), 0; sin(yaw), cos(yaw), 0; 0, 0, 1]; %z-axis
Rpitch = [cos(pitch), 0, sin(pitch); 0, 1, 0; -sin(pitch), 0, cos(pitch)]; %y-axis
Rroll = [1, 0, 0; 0, cos(roll), -sin(roll); 0, sin(roll), cos(roll)]; %x-axis

R = (Ryaw * Rpitch * Rroll);

d = 3;
c = [0; 0; d];
t = -R * c;

K = [674.9180, 0, 307.5513; ...
     0, 674.9180, 251.4542; ...
     0, 0, 1.0000];
   
P = K * [R, t];
Pcell = cell(1,1);
Pcell{1} = K \ P;
close all
figure
plot(X(1, :), X(2, :), '.');
hold on
plotcams(Pcell)
axis equal

%% Project into camera

x = P * X;
x = x ./ x(end, :);

figure
plot(x(1, :), x(2, :), '.')

%% Compute true hompgraphy

H = [P(:, 1:2), P(:, 4)];


%% Estimate homography

yaw = deg2rad(0); % assumed to be 0
pitch = deg2rad(truePitch); % known
roll = deg2rad(trueRoll); % known
Ryaw = [cos(yaw), -sin(yaw), 0; sin(yaw), cos(yaw), 0; 0, 0, 1]; %z-axis
Rpitch = [cos(pitch), 0, sin(pitch); 0, 1, 0; -sin(pitch), 0, cos(pitch)]; %y-axis
Rroll = [1, 0, 0; 0, cos(roll), -sin(roll); 0, sin(roll), cos(roll)]; %x-axis

Rhat = (Ryaw * Rpitch * Rroll);

dhat = 1;
chat = [0; 0; dhat];
tHat = -R * chat;
Phat = [Rhat, tHat];

Hhat = [Phat(:, 1:2), Phat(:, 4)];

Xhat = Hhat' * (K \ x);
Xhat = Xhat ./ Xhat(end, :);
figure
plot(Xhat(1, :), Xhat(2, :), '.');
axis equal