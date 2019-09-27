classdef horizonRegressionHuberPointsLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    properties
      
      x1
      x2
      yScale
      delta
    end
    
    methods
        function layer = horizonRegressionHuberPointsLayer(name, delta)
            % layer = horizonProductRegressionLayer(name) a layer for
            % regression on horizon lines
			
            % Set layer name.
            layer.Name = name;
            
            layer.delta = delta;

            % Set layer description.
            layer.Description = 'Calculates the normalized horizon loss';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) 
            
            y1hat = Y(1, 1, 1, :);
            y2hat = Y(1, 1, 2, :);
            
            y1 = T(1, 1, 1, :);
            y2 = T(1, 1, 2, :);
            
            d1 = (y1hat- y1).^2;
            d2 = (y2hat - y2).^2;
            
            d = max(cat(3, d1, d2), [], 3);
            d = layer.delta^2 * (sqrt(1 + (d / layer.delta^2)) - 1);
                
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
                dLdY(1, n) = d ./ sqrt((d / layer.delta).^2 + 1);
                dLdY(2, n) = 0;
              else
                d = d2(1, 1, 1, n);
                dLdY(1, n) = 0;
                dLdY(2, n) = d ./ sqrt((d / layer.delta).^2 + 1);
              end
            end
            dLdY = dLdY / N;
            dLdY = reshape(dLdY, [1, 1, 2, N]);
            dLdY = cast(dLdY, 'like', Y);
        end
    end
end