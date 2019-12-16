classdef smoothedCrossEntropyLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    properties
      
    end
    
    methods
        function layer = smoothedCrossEntropyLayer(name)
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Calculates the cross entropy loss for smooth labels';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) 

            N = size(Y, 4);
            loss = -T .* log(max(Y, 1e-15)) + T .* log(max(T, 1e-15));
            loss = sum(loss(:)) / N;
%             Nclasses = size(Y, 3) / 2;
% %             [~, maxRhoEst] = max(Y(1,1,1:Nclasses, :), [], 3);
% %             [~, maxRhoTrue] = max(T(1,1,1:Nclasses,:), [], 3);
% %             
% %             [~, maxThetaEst] = max(Y(1,1,Nclasses+1:end, :), [], 3);
% %             [~, maxThetaTrue] = max(T(1,1,Nclasses+1:end,:), [], 3);
%             
%             %loss1 = mean(abs(maxRhoEst - maxRhoTrue) < 10);
%             %loss2 = mean(abs(maxThetaEst - maxThetaTrue) < 10);
%             %loss = single((loss1 + loss2) / 2);
% %             loss = single(mean(maxThetaEst == maxThetaTrue));
% 
%             loss = -T(1,1,1:Nclasses,:) .* log(max(Y(1,1,1:Nclasses,:), 1e-15)) + ...
%                 T(1,1,1:Nclasses,:) .* log(max(T(1,1,1:Nclasses,:), 1e-15));
%             loss = sum(loss(:)) / Nclasses;
            
        end
        
        function dLdY = backwardLoss(layer, Y, T)

            N = size(Y, 4);
            dLdY = -T ./ max(Y, 1e-15) / N;
%             Nclasses = N/2;
%             dLdY(1,1,Nclasses+1:2*Nclasses,:) = 0;
        end
            
    end
end