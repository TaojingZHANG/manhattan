classdef argmaxLayer < nnet.layer.Layer
    % Returns the x and y coordinates of the maximum value for each channel

    properties
        % Layer learnable parameters
          
        % (none)
        tau % temperature

    end
    
    methods
      function layer = argmaxLayer(name, tau)
        
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
            Z = zeros(2, 1, M, N);
            Xe = exp(X);
            Ispace = linspace(-1, 1, I);
            Jspace = linspace(-1, 1, J);
            for n = 1:N
              for m = 1:M
                x = Xe(:, :, m, n);
                [~, maxInd] = max(x(:));
                [i, j] = ind2sub(size(x), maxInd);
                Z(:, :, m, n) = [Ispace(i); Jspace(j)];
              end
            end
            
            Z = cast(Z, 'like', X);
            
        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX1,…,dLdXn,dLdW] = backward(layer,X1,…,Xn,Z,dLdZ,~)
            % backward propagates the derivative of the loss function
            % through the layer.
            
            I = size(X, 1);
            J = size(X, 2);
            M = size(X, 3);
            N = size(X, 4);
            L = I * J;
%             onehot = zeros(size(X));
%             for n = 1:N
%               for m = 1:M
%                 image = X(:, :, m, n);
%                 [~, maxInd] = max(image(:));
%                 [i, j] = ind2sub(size(A), maxInd);
%                 onehot(i, j, m, n) = 1;
%               end
%             end
%             
            % Use Gumbel softmax trick for backpropagation

            Xe = exp(X);
            g = -log(-log(rand(size(Xe)))); 
            a = (log(Xe) + g) / layer.tau;
            dzdy1 = linspace(-1, 1, I);
            dzdy2 = linspace(-1, 1, J);
            %[i, j] = ind2sub(size(I), maxInd);
%             dZdX = zeros(2, 1, M, N);
            dLdX = zeros(size(Xe));
            for n = 1:N
              for m = 1:M
%                 tmp = Xexp(:, :, m, n);
%                 xi = tmp(:);
                tmp = a(:, :, m, n);
                ai = tmp(:);
                yi = softmax(ai);
                %yi = reshape(yi, size(image));
                %y(:, :, m, n) = yi;
                dyda = zeros(L, L);
                dadx = diag(ones(L, 1));
                for i = 1:L
                  for j = 1:L
                    if i == j
                      dyda(i, j) = yi(i) * (1 - yi(i));
                    else
                      dyda(i, j) = -yi(i) * yi(j);
                    end
                  end
                end
                dydx = sum(dyda .* dadx);
                dzdx1 = sum(dzdy1' * dydx, 2);
                dzdx2 = sum(dzdy2' * dydx, 2);
                dLdX(:, :, m, n) = dLdZ(1, 1, m, n) * dzdx1 + ...
                    dLdZ(2, 1, m, n) * dzdx2';
                
%                 for i = 1:I
%                   for j = 1:J
%                   dzdx1 = sum(dzdy1(i) * dydx);
%                   dzdx2 = sum(dzdy2(j) * dydx);
%                   dLdX(i, j, m, n) = dLdZ(1, 1, m, n) * dzdx1 + ...
%                     dLdZ(2, 1, m, n) * dzdx2;
%                   end
%                 end
%                 dZdX(:, :, m, n) = dzdx;
              end
            end
            
%             for i = 1:I
%               for j = 1:J
%                 dLdX(i, j, :, :) = dZdX(1, :, :, :) * dydz1(i) + ...
%                   dZdX(2, :, :, :) + dydz2(j);              end
%             end
            dLdX = cast(dLdX, 'like', X);

        end
    end
end