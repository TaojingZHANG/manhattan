classdef coordConvLayer < nnet.layer.Layer
    % Adds i and j coordinates as inputs before convolution

    properties
        % Layer learnable parameters
          
        % (none)

    end
    
    methods
      function layer = coordConvLayer(name)
        
        % Set layer name.
          layer.Name = name;

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            I = size(X, 1);
            J = size(X, 2);
            iCoord = repmat(1:I, [I, 1]);
            jCoord = repmat((1:J)', [1, J]);
            
            iCoord = iCoord / (I - 1);
            jCoord = jCoord / (J - 1);
            
            iCoord = iCoord * 2 - 1;
            jCoord = jCoord * 2 - 1;
            
            N = size(X, 3);
            L = size(X, 4);
            Z = zeros(I, J, N + 2, L); % The intersection (x, y)
            Z(:, :, 1:N, :) = X;
            Z(:, :, N + 1, :) = I;
            Z(:, :, N + 2, :) = J;

            Z = cast(Z, 'like', X);
            
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX1,…,dLdXn,dLdW] = backward(layer,X1,…,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.
            
            N = size(dLdZ, 3);
            dLdX = dLdZ(:, :, 1:N-2, :);
        end
    end
end