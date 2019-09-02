%% This script generates images of lines that intersect in mutually
% orthogonal vanishing points
rng(0);
close all
clear all


N = 1e3; %number of images
sigma2 = 0; %gaussian noise level

imres = [100, 100];

lineIms = single(zeros(imres(1), imres(2), 1, N)); % uint8
labels = zeros(1, 1, 3, N);
Nlines = [3, 5];

ploty = true; %vertical vp
plotx = false;
plotz = true;


figure;
for n = 1:N
  
  % Generate a random camera matrix: original viewing direction is along
  % positive z-axis of world coordinates (x=right,y=down)
    roll = 10 * randn() * pi / 180; % degrees
    pitch = 10 * randn() * pi / 180; % degrees
    yaw = 2 * pi * rand();
    
    Rroll = [cos(roll), -sin(roll), 0; ... % rotation about z-axis
            sin(roll),  cos(roll), 0; ...
            0, 0, 1];
    Ryaw = [cos(yaw), 0, sin(yaw); ... % rotation about y-axis
              0, 1, 0; ...
              -sin(yaw), 0, cos(yaw)];
    Rpitch= [1, 0, 0; ... % rotation about x-axis
             0, cos(pitch), -sin(pitch); ...
             0, sin(pitch), cos(pitch)];

    R = (Ryaw * Rpitch * Rroll)';

    vpx = R * [1; 0; 0];
    vpy = R * [0; 1; 0];
    vpz = R * [0; 0; 1];
    
    figure(1)
    hold off
    if ploty
      Nlinesy = randi(Nlines);
      for nline = 1:Nlinesy
        x1 = -1 + 2 * rand();
        P1 = [x1; 1];
        Vy = vpy(1:2) / vpy(3);
        l = (P1 - Vy) / norm(P1 - Vy);
        x2 = l(1) / l(2) * 2 + x1;
        P2 = [x2; -1];
        
        alpha1 = rand(); alpha2 = rand();
        p1 = P1 * alpha1 + P2 * (1 - alpha1);
        p2 = P1 * alpha2 + P2 * (1 - alpha2);
        
        plot([p1(1), p2(1)], [p1(2), p2(2)], 'w');
        hold on
      end
    end
    
    if plotx
      Nlinesx = randi(Nlines);
      for nline = 1:Nlinesx
        
      end
    end
    
     
    if plotz
      Nlinesz = randi(Nlines);
      for nline = 1:Nlinesz
        
        Vz = vpz(1:2) / vpz(3);
        inside = false;
        if Vz(1) > 1
          choice = 1;
        elseif Vz(1) < -1
          choice = 3;
        elseif Vz(2) > 1
          choice = 2;
        elseif Vz(2) < -1
          choice = 4;
        else
          choice = randi([1, 4]);
          inside = true;
        end
        
        if choice == 1 % horizontal lines come from the left
          y1 = -1 + 2 * rand();
          P1 = [-1; y1];
          l = (P1 - Vz) / norm(P1 - Vz);
          y2 = l(2) / l(1) * 2 + y1;
          P2 = [1; y2];
        elseif choice == 2 % bottom
          x1 = -1 + 2 * rand();
          P1 = [x1; -1];
          l = (P1 - Vz) / norm(P1 - Vz);
          x2 = l(1) / l(2) * 2 + x1;
          P2 = [x2; 1];
        elseif choice == 3 % right
          y1 = -1 + 2 * rand();
          P1 = [1; y1];
          l = (P1 - Vz) / norm(P1 - Vz);
          y2 = l(2) / l(1) * (-2) + y1;
          P2 = [-1; y2];
        else % top
          x1 = -1 + 2 * rand();
          P1 = [x1; 1];
          l = (P1 - Vz) / norm(P1 - Vz);
          x2 = l(1) / l(2) * (-2) + x1;
          P2 = [x2; -1];
        end
        
        if inside == true
          P2 = Vz;
        end
        
        alpha1 = rand(); alpha2 = rand();
        p1 = P1 * alpha1 + P2 * (1 - alpha1);
        p2 = P1 * alpha2 + P2 * (1 - alpha2);
        
        plot([p1(1), p2(1)], [p1(2), p2(2)], 'w');
        hold on
      end
    end

    
    
    axis([-1, 1, -1, 1])
    pbaspect([1 1 1])
    set(gca,'visible','off')
    set(gcf,'color','k');
    f = gcf;
    f.Position = [100, 100, imres(1), imres(2)];
    I = frame2im(getframe(f));
    Inoisy = imnoise(uint8(rgb2gray(I)), 'gaussian', 0, sigma2);
    lineIms(:, :, 1, n) = Inoisy';
    
    labels(:, :, :, n) = vpy * sign(vpy(3));
    hold off
    
end

%% Split train/test
ratio = 0.9;

train = 1:ceil(ratio * N);
test = ceil(ratio * N)+1:N;

trainIms = lineIms(:, :, :, train);
testIms = lineIms(:, :, :, test);

trainLabels = labels(:, :, :, train);
testLabels = labels(:, :, :, test);

%% Save images and labels
save('vpData', 'trainIms', 'testIms', 'trainLabels', 'testLabels');