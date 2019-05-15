classdef twoLineLayer < nnet.layer.Layer
    % Two line layer.
    

    properties
        % Layer learnable parameters
          
        % (none)

    end
    
    methods
      function layer = twoLineLayer(name)
        % layer = twoLineLayer creates a layer that takes
        % two lines a1 * x + b1 * y + c1 = 0 and a2 * x + b2 * y + c2 =
        % 0 and calculates the intersection.
        
        % Set layer name.
          layer.Name = name;

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            assert(size(X, 3) == 6, 'Last layer before regression must have 6 nodes');
            N = size(X, 4);
            a1 = (X(:, :, 1, :));
            b1 = (X(:, :, 2, :));
            c1 = (X(:, :, 3, :));
            
            a2 = (X(:, :, 4, :));
            b2 = (X(:, :, 5, :));
            c2 = (X(:, :, 6, :));
            
            alpha = b1 .* c2 - c1 .* b2;
            beta = a2 .* c1 - a1 .* c2;
            gamma = a1 .* b2 - b1 .* a2;
            
            x = alpha ./ gamma;
            y = beta ./ gamma;
            
            Z = (zeros(1, 1, 2, N)); % The intersection (x, y)
            Z(1, 1, 1, :) = (x);
            Z(1, 1, 2, :) = (y);
            Z = cast(Z, 'like', X);
            
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX1,…,dLdXn,dLdW] = backward(layer,X1,…,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.
            
             % Layer input size must corresond to two lines
            assert(size(X, 3) == 6, 'Last layer before regression must have 6 nodes');
            assert(size(dLdZ, 3) == 2, 'Num outputs must be 2');
            N = size(X,4); % minibatch size

            a1 = (X(:, :, 1, :));
            b1 = (X(:, :, 2, :));
            c1 = (X(:, :, 3, :));
            
            a2 = (X(:, :, 4, :));
            b2 = (X(:, :, 5, :));
            c2 = (X(:, :, 6, :));
            
            alpha = b1 .* c2 - c1 .* b2;
            beta = a2 .* c1 - a1 .* c2;
            gamma = a1 .* b2 - b1 .* a2;
            
            dxda1 = -(alpha .* b2) ./ gamma.^2;
            dxda2 = (alpha .* b1) ./ gamma.^2;
            dxdb1 = -(beta .* b2) ./ gamma.^2;
            dxdb2 = (beta .* b1) ./ gamma.^2;
            dxdc1 = -b2 ./ gamma; 
            dxdc2 = b1 ./ gamma;
            
            dyda1 = (alpha .* a2) ./ gamma.^2;
            dyda2 = -(alpha .* a1) ./ gamma.^2;
            dydb1 = (beta .* a2) ./ gamma.^2;
            dydb2 = -(beta .* a1) ./ gamma.^2;
            dydc1 = a2 ./ gamma;
            dydc2 = -a1 ./ gamma;
            
            dldx = dLdZ(1, 1, 1, :);
            dldy = dLdZ(1, 1, 2, :);
            
            dlda1 = dldx .* dxda1 + dldy .* dyda1;
            dlda2 = dldx .* dxda2 + dldy .* dyda2;
            dldb1 = dldx .* dxdb1 + dldy .* dydb1;
            dldb2 = dldx .* dxdb2 + dldy .* dydb2;
            dldc1 = dldx .* dxdc1 + dldy .* dydc1;
            dldc2 = dldx .* dxdc2 + dldy .* dydc2;
            
            dLdX = zeros(size(X));
            dLdX(1, 1, 1, :) = dlda1;
            dLdX(1, 1, 2, :) = dldb1;
            dLdX(1, 1, 3, :) = dldc1;
            dLdX(1, 1, 4, :) = dlda2;
            dLdX(1, 1, 5, :) = dldb2;
            dLdX(1, 1, 6, :) = dldc2;
                        
            dLdX = cast(dLdX, 'like', X);
        end
    end
end