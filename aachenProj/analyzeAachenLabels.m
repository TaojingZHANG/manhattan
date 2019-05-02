load('aachenDs.mat');

maxRoll = 0;
maxPitch = 0;
rolls = zeros(length(aachenDsTrain.Labels), 1);
pitchs = zeros(length(aachenDsTrain.Labels), 1);
for i = 1:length(aachenDsTrain.Labels)
  tmp = squeeze(aachenDsTrain.Labels{i});
  
  f = aachenDsTrain.Intrinsics{4}(i);
  cx = aachenDsTrain.Intrinsics{5}(i);
  cy = aachenDsTrain.Intrinsics{6}(i);
  
  K = [f, 0, cx; 0, f, cy; 0, 0, 1];
  
  tmp = [tmp; 1];
  tmp = K \ tmp;
  tmp = tmp / norm(tmp); % [x, y, z] --> [z, x, y]
  pitch = atan(-tmp(3) / sqrt(tmp(1)^2 + tmp(2)^2));
  roll = atan(tmp(1) / tmp(2)); % (90 degrees?)
  if abs(roll) > abs(maxRoll)
    maxRoll = roll;
    maxRollIm = aachenDsTrain.Datastore.Files{i};
    maxRolli = i;
  end
  if abs(pitch) > abs(maxPitch)
    maxPitch = pitch;
    maxPitchIm = aachenDsTrain.Datastore.Files{i};
    maxPitchi = i;
  end
  rolls(i) = roll;
  pitchs(i) = pitch;
end

close all
subplot(211)
histogram(rad2deg(pitchs))
xlabel('Pitch [degrees]')
ylabel('Frequency')
subplot(212)
histogram(rad2deg(rolls))
xlabel('Roll [degrees')
ylabel('Frequency')
