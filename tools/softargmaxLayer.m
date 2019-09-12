classdef indexLayer < nnet.layer.Layer
    % Used as spatial softmax if placed after softmax layer

    properties
        % Layer learnable parameters

    end
    
    methods
      function layer = indexLayer(name)
        
        % Set layer name.
          layer.Name = name;

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            I = size(X, 1);
            J = size(X, 2);
            M = size(X, 3);
            N = size(X, 4);
            Z = zeros(I, J, 2 * M, N);
            xspace = linspace(-1, 1, I);
            yspace = linspace(-1, 1, J);
            
            [Xspace,Yspace] = meshgrid(xspace,yspace);

            for m = 1:M
              Z(:, :, 2 * m - 1, :) = X(:, :, m, :) .* Xspace;
              Z(:, :, 2 * m, :) = X(:, :, m, :) .* Yspace;
            end
            
            Z = cast(Z, 'like', X);
            
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX1,…,dLdXn,dLdW] = backward(layer,X1,…,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.
            
            I = size(X, 1);
            J = size(X, 2);
            M = size(X, 3);
            N = size(X, 4);
            L = I * J;

            dLdX = zeros(size(X));

            xspace = linspace(-1, 1, I);
            yspace = linspace(-1, 1, J);
            
            [Xspace,Yspace] = meshgrid(xspace,yspace);
            
            for m = 1:M
              dLdX(:, :, m, :) = dLdZ(:, :, 2 * m - 1, :) .* Xspace + ...
                dLdZ(:, :, 2 * m , :) .* Yspace;
            end
            
            dLdX = cast(dLdX, 'like', X);

        end
    end
end