classdef customSoftmaxLayer < nnet.layer.Layer
    methods
        function layer = customSoftmaxLayer(name)
            % Set layer name
            layer.Name = name;
            % Set layer description
            layer.Description = 'customSoftmaxLayer';
        end
        function Z = predict(layer,X)
            % Forward input data through the layer and output the result
            X = X - max(X, [], 3);
            Z = exp(X) ./ sum(exp(X), 3);
        end
        function dLdX = backward(layer, X ,Z,dLdZ, ~)
            % Backward propagate the derivative of the loss function through
            % the layer
            N = size(Z, 4);
            M = size(Z, 3);
            dLdZ = reshape(dLdZ, [N, M]);
            Z = reshape(Z, [N, M]);
            dLdX = Z .* (dLdZ - (dLdZ .* Z) * ones(M, M));
            dLdX = reshape(dLdX, [1, 1, M, N]);
            
        end
    end
end