N = 1000;
sigma2 = 0.01;

imres = [78, 78];
lineIms = uint8(zeros(imres(1), imres(2), 1, N));
labels = zeros(1, N);

close all
f = figure('Position', [100 100 100 100]);
for n = 1:N
  for theta = 0:5:360
    p1 = [0; 0];
    r = 1;
    x = p1(1) + r * cosd(theta);
    y = p1(2) + r * sind(theta);
    p2 = [x; y];
  
    figure(f)
    ax = subplot(1,1,1);
    plot([p1(1), p2(1)], [p1(2), p2(2)], 'w');
    
    axis([-1, 1, -1, 1])
    pbaspect([1 1 1])
    set(gca,'visible','off')
    set(gcf,'color','k');
    I = rgb2gray(frame2im(getframe(ax)));
    
    X = log(fftshift(abs(fft2(I))));
    figure(2)
    imagesc(X)
    colormap gray
    waitforbuttonpress
    
  end
  
end
