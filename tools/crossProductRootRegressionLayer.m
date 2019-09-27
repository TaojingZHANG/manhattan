classdef crossProductRootRegressionLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    properties
      k
    end
    
    methods
        function layer = crossProductRootRegressionLayer(name, k)
            % layer = crossProductRegressionLayer(name) a layer for
            % regression in the projective plane
			
            % Set layer name.
            layer.Name = name;
            layer.k = k; %kth root

            % Set layer description.
            layer.Description = 'Calculates the loss (|a x b|^2)^(1/k)';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the cross product loss between
            % the predictions Y and the training targets T.

            assert(size(Y, 3) == 3, 'Input must be 3D vector');
            
            crossLoss = (sum(cross(Y, T, 3).^2, 3)).^(1/layer.k);
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = 0.5 * sum(crossLoss) / N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the cross loss with respect to the predictions Y

            N = size(Y,4);
            
            a1 = Y(:, :, 1, :);
            a2 = Y(:, :, 2, :);
            a3 = Y(:, :, 3, :);
            
            b1 = T(:, :, 1, :);
            b2 = T(:, :, 2, :);
            b3 = T(:, :, 3, :);
            
            dLda1 = (a3 .* b1 - a1 .* b3) .* (-b3) + ...
              (a1 .* b2 - a2 .* b1) .* b2;
            dLda2 = (a2 .* b3 - a3 .* b2) .* b3 + ...
              (a1 .* b2 - a2 .* b1) .* (-b1);
            dLda3 = (a2 .* b3 - a3 .* b2) .* (-b2) + (a3 .* b1 - a1 .* b3) .* b1;
            
            dLdX = cat(3, dLda1, dLda2, dLda3);
            
            crossLoss = (sum(cross(Y, T, 3).^2, 3));
            
            dLdY = dLdX .* crossLoss.^(1/layer.k - 1) / layer.k;
            
            dLdY = dLdY / N;
        end
    end
end