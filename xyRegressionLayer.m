classdef xyRegressionLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    methods
        function layer = xyRegressionLayer(name)
            % layer = xyRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Two output MSE regression layer';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the xy loss between
            % the predictions Y and the training targets T.

            % Calculate xy.
            R = size(Y,3);
            meanSquareError = sum((Y-T).^2, 3)/R;
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = 0.5 * sum(meanSquareError)/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the xy loss with respect to the predictions Y

            R = size(Y,3);
            N = size(Y,4);
            dLdY = (Y-T) / (N*R);
        end
    end
end