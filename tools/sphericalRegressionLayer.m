classdef sphericalRegressionLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    properties
      epsilon; 
    end
    
    methods
        function layer = sphericalRegressionLayer(name, epsilon)
			
            % Set layer name.
            layer.Name = name;
            
            % Set epsilon for numerical stability
            layer.epsilon = epsilon;

            % Set layer description.
            layer.Description = 'Regression layer with spherical loss function';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the xy loss between
            % the predictions Y and the training targets T.
            
            assert(size(Y, 3) == 3, ...
              'Last layer before regression must have 3 nodes for spherical regression');

            % Calculate xy.
            R = size(Y,3);
            Ynorm = Y ./ sqrt(sum(Y.^2, 3) + layer.epsilon);
            Tnorm = T ./ sqrt(sum(T.^2, 3) + layer.epsilon);
            meanSquareError = sum((Ynorm - Tnorm).^2, 3) / R;
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = 0.5 * sum(meanSquareError)/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the xy loss with respect to the predictions Y
            
            R = size(Y,3);
            N = size(Y,4);
            
            dLdY = zeros(size(Y));
            
            x = Y(1, 1, 1, :);
            y = Y(1, 1, 2, :);
            z = Y(1, 1, 3, :);
            
            scaleDiv = sqrt(x.^2 + y.^2 + z.^2) + layer.epsilon;
            Tnorm = T ./ sqrt(sum(T.^2, 3) + layer.epsilon);
            
            xt = Tnorm(1, 1, 1, :);
            yt = Tnorm(1, 1, 2, :);
            zt = Tnorm(1, 1, 3, :);
            
            dLdx = (-xt .* (y.^2 + z.^2) + yt .* x .* y + zt .* x .* z) ./ ...
              (scaleDiv.^3);
            dLdy = (-yt .* (z.^2 + x.^2) + zt .* y .* z + xt .* y .* x) ./ ...
              (scaleDiv.^3);
            dLdz = (-zt .* (x.^2 + y.^2) + xt .* z .* x + yt .* z .* y) ./ ...
              (scaleDiv.^3);
           
            
            dLdY(1, 1, 1, :) = dLdx / (N * R);
            dLdY(1, 1, 2, :) = dLdy / (N * R);
            dLdY(1, 1, 3, :) = dLdz / (N * R);
            
            dLdY = cast(dLdY, 'like', Y);


        end
    end
end