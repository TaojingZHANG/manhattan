classdef horizonRegressionAtanLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    properties
      
      x1
      x2
    end
    
    methods
        function layer = horizonRegressionAtanLayer(name, x1, x2)
            % layer = horizonProductRegressionLayer(name) a layer for
            % regression on horizon lines
			
            % Set layer name.
            layer.Name = name;
            
            layer.x1 = x1;
            layer.x2 = x2;

            % Set layer description.
            layer.Description = 'Calculates the normalized horizon loss';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) 

            assert(size(Y, 3) == 3, 'Input must be 3D vector');
            
            ahat = Y(1, 1, 1, :);
            bhat = Y(1, 1, 2, :);
            chat = Y(1, 1, 3, :);
            
            a = T(1, 1, 1, :);
            b = T(1, 1, 2, :);
            c = T(1, 1, 3, :);
            
            y1 = (a * layer.x1 + c) ./ b;
            y2 = (a * layer.x2 + c) ./ b;
            
            y1hat = (ahat * layer.x1 + chat) ./ bhat;
            y2hat = (ahat * layer.x2 + chat) ./ bhat;
            
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
            
            
            ahat = Y(1, 1, 1, :);
            bhat = Y(1, 1, 2, :);
            chat = Y(1, 1, 3, :);
            
            a = T(1, 1, 1, :);
            b = T(1, 1, 2, :);
            c = T(1, 1, 3, :);
            
            y1 = (a * layer.x1 + c) ./ b;
            y2 = (a * layer.x2 + c) ./ b;
            
            y1hat = (ahat * layer.x1 + chat) ./ bhat;
            y2hat = (ahat * layer.x2 + chat) ./ bhat;
            
            d1 = (y1hat - y1);
            d2 = (y2hat - y2);
            
            dLdY = (zeros(3, N));
            
            for n = 1:N
              if d1(1, 1, 1, n)^2 > d2(1, 1, 1, n)^2
                  d = d1(1, 1, 1, n);
                  dLdY(1, n) = sign(d) / (1 + d^2) * layer.x1 / bhat(1, 1, 1, n);
                  dLdY(2, n) = -sign(d) / (1 + d^2) * (ahat(1, 1, 1, n) * layer.x1 + chat(1, 1, 1, n)) / bhat(1, 1, 1, n)^2;
                  dLdY(3, n) = sign(d) / (1 + d^2) * 1 / bhat(1, 1, 1, n);
              else
                  d = d2(1, 1, 1, n);
                  dLdY(1, n) = sign(d) / (1 + d^2) * layer.x2 / bhat(1, 1, 1, n);
                  dLdY(2, n) = -sign(d) / (1 + d^2) * (ahat(1, 1, 1, n) * layer.x2 + chat(1, 1, 1, n)) / bhat(1, 1, 1, n)^2;
                  dLdY(3, n) = sign(d) / (1 + d^2) * 1 / bhat(1, 1, 1, n);
              end
            end
            dLdY = dLdY / N;
            dLdY = reshape(dLdY, [1, 1, 3, N]);
            dLdY = cast(dLdY, 'like', Y);
        end
    end
end