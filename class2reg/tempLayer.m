classdef tempLayer < nnet.layer.Layer

    properties
        % Layer learnable parameters
          
        % (none)
        temp

    end
    
    methods
      function layer = tempLayer(name, temp)
        
        % Set layer name.
          layer.Name = name;
          layer.temp = temp;

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            Z = X / layer.temp;
            
        end
        
        function [dLdX] = backward(layer, ~, ~, dLdZ, ~)
            % [dLdX1,???,dLdXn,dLdW] = backward(layer,X1,???,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.
            
            dLdX = dLdZ / layer.temp;
            
        end
    end
end