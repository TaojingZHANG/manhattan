classdef binaryCrossEntropyLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    properties
      
    end
    
    methods
        function layer = binaryCrossEntropyLayer(name)
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Calculates the cross entropy loss for smooth labels';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) 

            N = size(Y, 4);
            loss = -(T .* log(max(Y, 1e-15)) + (1 - T) .* log(max(1 - Y, 1e-15)));
            loss = sum(loss(:)) / N;
        
        end
        
%         function dLdY = backwardLoss(layer, Y, T)
% 
%             N = size(Y, 4);
%             dLdY = (Y - T) ./ (max(Y, 1e-15) .* (1 - max(Y, 1e-15)));
%         end
            
    end
end