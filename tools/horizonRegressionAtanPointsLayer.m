classdef horizonRegressionAtanPointsLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    properties
      
      x1
      x2
    end
    
    methods
        function layer = horizonRegressionAtanPointsLayer(name, x1, x2)
            % layer = horizonProductRegressionLayer(name) a layer for
            % regression on horizon lines
			
            % Set layer name.

            % Set layer description.
            layer.Description = 'Calculates the normalized horizon loss';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) 

            
            y1hat = Y(1, 1, 1, :);
            y2hat = Y(1, 1, 2, :);
            
            y1 = T(1, 1, 1, :);
            y2 = T(1, 1, 2, :);
            
            d1 = abs(y1hat- y1);
            d2 = abs(y2hat - y2);
            
            d = max(cat(3, d1, d2), [], 3);
            
            d = atan(d);
                
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = sum(d) / N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the horizon loss with respect to the predictions Y

            N = size(Y,4);
            
            
            y1hat = Y(1, 1, 1, :);
            y2hat = Y(1, 1, 2, :);
            
            y1 = T(1, 1, 1, :);
            y2 = T(1, 1, 2, :);
            
            
            d1 = (y1hat - y1);
            d2 = (y2hat - y2);
            
            dLdY = (zeros(2, N));
            
            for n = 1:N
              if d1(1, 1, 1, n)^2 > d2(1, 1, 1, n)^2
                  d = d1(1, 1, 1, n);
                  dLdY(1, n) = sign(d) / (1 + d^2);
                  dLdY(2, n) = 0;
              else
                  d = d2(1, 1, 1, n);
                  dLdY(1, n) = 0;
                  dLdY(2, n) = sign(d) / (1 + d^2);
              end
            end
            dLdY = dLdY / N;
            dLdY = reshape(dLdY, [1, 1, 2, N]);
            dLdY = cast(dLdY, 'like', Y);
        end
    end
end