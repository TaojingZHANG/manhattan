classdef cosineSphereLayer < nnet.layer.Layer
    % Adds i and j coordinates as inputs before convolution

    properties
        % Layer learnable parameters
          
        % (none)

    end
    
    methods
      function layer = cosineSphereLayer(name)
        
        % Set layer name.
          layer.Name = name;

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            r = X(:, :, 1, :);
            theta = X(:, :, 2, :);
            f1 = exp(r) .* cos(theta);
            f2 = exp(r) .* sin(theta);
            f = cat(3, f1, f2);
            fa = cat(3, f, ones(size(r)));
            A = sqrt(sum(fa.^2, 3));
            Z = f ./ A;

            Z = cast(Z, 'like', X);
            
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX1,???,dLdXn,dLdW] = backward(layer,X1,???,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.
            dLdX = zeros(size(X));
            r = X(:, :, 1, :);
            theta = X(:, :, 2, :);
            f1 = exp(r) .* cos(theta);
            f2 = exp(r) .* sin(theta);
            f = cat(3, f1, f2);
            fa = cat(3, f, ones(size(r)));
            A = sqrt(sum(fa.^2, 3));
            Z = f ./ A;
            dLdX = cast(dLdX, 'like', dLdZ);
            
            z1 = Z(:, :, 1, :);
            z2 = Z(:, :, 2, :);
            
            dz1dx1 = z1 .* (1 - z1.^2 - z2.^2); %dp1do1
            dz1dx2 = -z2; % dp1po2
            
            dz2dx1 = z2 .* (1 - z1.^2 - z2.^2); % dp2do1
            dz2dx2 = z1; % dp2do2
            
            dLdX(:, :, 1, :) = dLdZ(:, :, 1, :) .* dz1dx1 + dLdZ(:, :, 2, :) .* dz2dx1;
            dLdX(:, :, 2, :) = dLdZ(:, :, 1, :) .* dz1dx2 + dLdZ(:, :, 2, :) .* dz2dx2;
            
        end
    end
end