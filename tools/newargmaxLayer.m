classdef newargmaxLayer < nnet.layer.Layer
    % Returns the x and y coordinates of the maximum value for each channel

    properties
        % Layer learnable parameters
          
        % (none)
        tau % temperature

    end
    
    methods
      function layer = newargmaxLayer(name, tau)
        
        % Set layer name.
          layer.Name = name;
          layer.tau = tau; %temperature

        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            
            I = size(X, 1);
            J = size(X, 2);
            M = size(X, 3);
            N = size(X, 4);
            Ispace = linspace(-1, 1, I);
            Jspace = linspace(-1, 1, J);
            
            Xs = reshape(X, [I*J, M, N]);
            [~, maxInd] = max(Xs, [], 1);
            [i, j] = ind2sub([I, J, M, N], maxInd);
            Z = reshape([Ispace(i); Jspace(j)], [2, 1, M, N]);
            
            Z = cast(Z, 'like', X);
            
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX1,???,dLdXn,dLdW] = backward(layer,X1,???,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.                        
            
            I = size(X, 1);
            J = size(X, 2);
            M = size(X, 3);
            N = size(X, 4);
            L = I * J;
            
            Xs = reshape(X, [I*J, M, N]);
            [~, maxInd] = max(Xs, [], 1);
            [indI, indJ] = ind2sub([I, J, M, N], maxInd);
            Xind = zeros(size(Xs));
            Xind(maxInd) = 1;
            Xind = reshape(Xind, [I, J, M, N]);
            Z = reshape([indI; indJ], [2, 1, M, N]);

            Xe = exp(X);
            g = -log(-log(rand(size(Xe)))); 
            a = (log(Xe) + g) / layer.tau;
            dzdy1 = linspace(-1, 1, I);
            dzdy2 = linspace(-1, 1, J);
            dLdX = zeros(size(X));
            dLdX = cast(dLdX, 'like', X);

            for n = 1:N
              for m = 1:M
                tmp = a(:, :, m, n);
                ai = tmp(:);
                yi = softmax(ai);
                dadx = ones(L, 1);
                dyda = yi.*(1-yi);
                dydx = dyda .* dadx;
                curri = indI(1, m, n);
                currj = indJ(1, m, n);
                dzdx1 = sum(dzdy1(curri) * dydx);
                dzdx2 = sum(dzdy2(currj) * dydx);

                dLdX(curri, currj, m, n) = dLdZ(1, 1, m, n) * dzdx1 + ...
                    dLdZ(2, 1, m, n) * dzdx2;

              end
            end
            
        end
    end
end