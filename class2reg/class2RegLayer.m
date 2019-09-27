classdef class2RegLayer < nnet.layer.Layer
    % Adds i and j coordinates as inputs before convolution

    properties
        % Layer learnable parameters
          
        % (none)
        weights

    end
    
    methods
      function layer = class2RegLayer(name, weights)
        
        % Set layer name.
          layer.Name = name;
          layer.weights = weights;

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            assert(size(X, 1) == 1);
            assert(size(X, 2) == 1);
            
            Ztemp = layer.weights * reshape(X, [size(X, 3), size(X, 4)]);
            Z = reshape(Ztemp, [1, 1, size(Ztemp, 1), size(Ztemp, 2)]);
            
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX1,???,dLdXn,dLdW] = backward(layer,X1,???,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.
            
            dLdX = reshape(layer.weights' * ...
              reshape(dLdZ, size(dLdZ, 3), size(dLdZ, 4)), size(X));
            
        end
    end
end