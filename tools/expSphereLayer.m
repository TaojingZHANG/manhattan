classdef expSphereLayer < nnet.layer.Layer
    % Adds i and j coordinates as inputs before convolution

    properties
        % Layer learnable parameters
          
        % (none)

    end
    
    methods
      function layer = expSphereLayer(name)
        
        % Set layer name.
          layer.Name = name;

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            f = exp(X);
            Z = f ./ sqrt(sum(f.^2, 3));

            Z = cast(Z, 'like', X);
            
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX1,???,dLdXn,dLdW] = backward(layer,X1,???,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.
            
            f = exp(X);
            Z = f ./ sqrt(sum(f.^2, 3));
            dLdX = zeros(size(Z));
            dLdX = cast(dLdX, 'like', dLdZ);
            for j = 1:3
              for i = 1:3
                if i == j
                  pi = Z(:, :, i, :);
                  dLdX(:, :, i, :) = dLdX(:, :, i, :) + dLdZ(:, :, j, :) .* pi .* (1 - pi.^2);
                else
                  pi = Z(:, :, i, :);
                  pj = Z(:, :, j, :);
                  dLdX(:, :, i, :) = dLdX(:, :, i, :) - dLdZ(:, :, j, :) .* pi.^2 .* pj;
                end
              end
            end
        end
    end
end