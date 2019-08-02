classdef sphereLayer < nnet.layer.Layer
    % Adds i and j coordinates as inputs before convolution

    properties
        % Layer learnable parameters
          
        % (none)

    end
    
    methods
      function layer = sphereLayer(name)
        
        % Set layer name.
          layer.Name = name;

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            %X(:, :, 3, :) = 0;
            Z = X ./ sqrt(sum(X.^2, 3));

            Z = cast(Z, 'like', X);
            
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX1,???,dLdXn,dLdW] = backward(layer,X1,???,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.
            dLdX = zeros(size(X));
            A = sqrt(sum(X.^2, 3));
            Z = X ./ A;
            dLdX = cast(dLdX, 'like', dLdZ);
            for j = 1:3
              for i = 1:3 %2
                if i == j
                  pi = Z(:, :, i, :);
                  dLdX(:, :, i, :) = dLdX(:, :, i, :) + dLdZ(:, :, j, :) .* (1 - pi.^2) ./ A;
                else
                  pi = Z(:, :, i, :);
                  pj = Z(:, :, j, :);
                  dLdX(:, :, i, :) = dLdX(:, :, i, :) - dLdZ(:, :, j, :) .* pi .* pj ./ A;
                end
              end
            end
        end
    end
end