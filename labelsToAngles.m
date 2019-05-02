function angles = labelsToAngles(labels, camParams, mu, sigma)

angles = zeros(length(labels), 2);
for i = 1:length(labels)
    
    % Undo normalization by mean and std
    label = labels(i, :);
    vp = zeros(3, 1);
    vp(1) = label(1) * sigma(1) + mu(1);
    vp(2) = label(2) * sigma(2) + mu(2);
    vp(3) = 1;
    
    % Normalize by camera matrix
    f = camParams{4}(i);
    cx = camParams{5}(i);
    cy = camParams{6}(i);
  
    K = [f, 0, cx; 0, f, cy; 0, 0, 1];
    
    vpCal = K \ vp;
    vpCal = vpCal / norm(vpCal);
    
    pitch = atan(-vpCal(3) / sqrt(vpCal(1)^2 + vpCal(2)^2));
    roll = atan(vpCal(1) / vpCal(2));
    
    angles(i, :) = [pitch, roll];
    
end
    
end

