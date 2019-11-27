classdef sordDataStore < matlab.io.Datastore & ...
                       matlab.io.datastore.MiniBatchable & ...
                       matlab.io.datastore.Shuffleable & ...
                       matlab.io.datastore.PartitionableByIndex
    
    properties
        Datastore
        Labels
        MiniBatchSize
        imSize
        croppedSize
        horizontalFlip
        randomCrop
        meanImage
        rhoClasses
        thetaClasses
        bw % use black and white?
        NumClasses
        thetaScale
        rhoScale
    end
    
    properties(SetAccess = protected)
        NumObservations
    end

    properties(Access = private)
        % This property is inherited from Datastore
        CurrentFileIndex
    end


    methods
        
        function ds = sordDataStore(imDs, inputLabels, imSize, croppedSize, horizontalFlip, ...
            rhoClasses, thetaClasses, meanImage, bw, rhoScale, thetaScale)

            % Create a file datastore. The readSequence function is
            % defined following the class definition.
%             fds = fileDatastore(folder, ...
%                 'ReadFcn',@imread, ...
%                 'IncludeSubfolders',true);
            ds.Datastore = imDs;

            % Read labels from folder names
            numObservations = numel(imDs.Files);
            labels = cell(length(inputLabels), 1);
            for i = 1:numObservations
                labels{i,1} = reshape(inputLabels(:, i), [1, 1, size(inputLabels, 1)]);
            end
            ds.Labels = labels;
            
            % Initialize datastore properties.
            ds.MiniBatchSize = 128;
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;
            ds.imSize = imSize;
            ds.croppedSize = croppedSize;
            ds.horizontalFlip = horizontalFlip;
            ds.rhoClasses = rhoClasses;
            ds.thetaClasses = thetaClasses;
            ds.meanImage = meanImage;
            ds.bw = bw;
            if nargin < 10
              ds.rhoScale = 1;
            else
              ds.rhoScale = rhoScale;
            end
            if nargin < 11
              ds.thetaScale = 1;
            else
              ds.thetaScale = thetaScale;
            end
            
        end
        
        
        function subds = partitionByIndex(myds, indices)
            subds = copy(myds);
            subds.Datastore = copy(myds.Datastore);
            subds.Datastore.Files = myds.Datastore.Files(indices);
            subds.Labels = myds.Labels(indices);
            subds.NumObservations = length(indices);
            reset(subds);
        end

        function tf = hasdata(ds)
            % Return true if more data is available
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end

        function [data,info] = read(ds)            
            % Read one mini-batch batch of data
            miniBatchSize = ds.MiniBatchSize;
            info = struct;
            
            fileIndices = zeros(miniBatchSize, 1);
            for i = 1:miniBatchSize
                predictors{i,1} = read(ds.Datastore);
                responses(i,1) = ds.Labels(ds.CurrentFileIndex);
                fileIndices(i) = ds.CurrentFileIndex;
                ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            end
            
            data = preprocessData(ds,predictors,responses, fileIndices);
        end

        function data = preprocessData(ds,predictors,responses, fileIndices)
            % data = preprocessData(ds,predictors,responses) preprocesses
            % the data in predictors and responses and returns the table
            % data
            
            miniBatchSize = ds.MiniBatchSize;
            
            % Subtract mean image, crop image to correct imagesize and make grayscale
            for i = 1:miniBatchSize
                X = predictors{i};
                l = responses{i};
                % horizontal flip
                if ds.horizontalFlip
                  if rand()
                    X = flip(X,2);
                    x1 = l(1); y1 = l(2);
                    x2 = l(3); y2 = l(4);
                    
                    l = [x1, y2, x2, y1];
                  end
                end
                
                origSize = size(X);
                if ds.randomCrop
                  smallSize = ds.imSize;
                  if origSize(1) > origSize(2) % vertical
                    newSize = [NaN, smallSize];
                    scaleFactor = origSize(2) / smallSize;
                  elseif origSize(2) > origSize(1) % horizontal
                    newSize = [smallSize, NaN];
                    scaleFactor = origSize(1) / smallSize;
                  else % square
                    newSize = [smallSize, smallSize];
                    scaleFactor = origSize(1) / smallSize;
                  end
                else
                  newSize = [ds.croppedSize, ds.croppedSize];
                  if origSize(1) > origSize(2) % vertical
                    scaleFactor = origSize(2) / ds.croppedSize;
                    d = origSize(1) - origSize(2);
                    ii = (floor(d/2)+1):(origSize(2) + floor(d/2));
                    X = X(ii, :, :);
                  elseif origSize(2) > origSize(1) % horizontal
                    scaleFactor = origSize(1) / ds.croppedSize;
                    d = origSize(2) - origSize(1);
                    ii = (floor(d/2)+1):(origSize(1) + floor(d/2));
                    X = X(:, ii, :);
                  else % square
                    scaleFactor = origSize(1) / ds.croppedSize;
                  end
                end
                
                Xsmall = imresize(X, newSize);
                
                if ds.randomCrop
                  % Extract random crop
                  L = ds.croppedSize(1);
                  sz = size(Xsmall);
                  n = sz(1); m = sz(2);
                  xr = randi(n-L+1);
                  yr = randi(m-L+1);
                  Xcrop = Xsmall(xr+(0:L-1),yr+(0:L-1), :);
                  
                  % calculate new origo
                  newOrigo = [xr - (n-L)/2; yr - (m-L)/2];
                else
                  Xcrop = Xsmall;
                  newOrigo = [0, 0];
                end

                % calculate rho and theta
                x1New = l(1) - newOrigo(1); y1New = l(2) - newOrigo(2);
                x2New = l(3) - newOrigo(1); y2New = l(4) - newOrigo(2);
                
                rho = (x2New * y1New - y2New * x1New) / sqrt( (y2New-y1New)^2 + (x2New - x1New)^2) / scaleFactor;
                theta = atand((y2New-y1New) / (x2New - x1New));
                
                
                % calculate SORD labels
                phiRho = @(x,y) (x-y).^2 / ds.rhoScale;
                phiTheta = @(x,y) (x-y).^2 / ds.thetaScale;
                rhoLabels = createSordLabels(rho, ds.rhoClasses, phiRho);
                thetaLabels = createSordLabels(theta, ds.thetaClasses, phiTheta);
                
                responses{i} = cat(3, rhoLabels, thetaLabels);
                
                
%                 if ds.bw
%                   predictors{i} = single(rgb2gray(Xcrop)) - single(ds.meanImage);
%                 else
%                   predictors{i} = single(Xcrop) - single(ds.meanImage);
%                 end
                %predictors{i} = predictors{i} / 127; % normalize to [-1, 1]
                predictors{i} = single(Xcrop);
            end
            
            % Return data as a table.
            data = table(predictors,responses);
        end

        function reset(ds)
            % Reset to the start of the data
            reset(ds.Datastore);
            ds.CurrentFileIndex = 1;
        end
        
        
        function dsNew = shuffle(ds)
            % dsNew = shuffle(ds) shuffles the files and the
            % corresponding labels in the datastore.
            
            % Create a copy of datastore
            dsNew = copy(ds);
            dsNew.Datastore = copy(ds.Datastore);
            fds = dsNew.Datastore;
            
            % Shuffle files and corresponding labels
            numObservations = dsNew.NumObservations;
            idx = randperm(numObservations);
            fds.Files = fds.Files(idx);
            dsNew.Labels = dsNew.Labels(idx);
            
        end
        
    end 

    methods (Hidden = true)

        function frac = progress(ds)
            % Determine percentage of data read from datastore
            frac = (ds.CurrentFileIndex - 1) / ds.NumObservations;
        end

    end
    
%     methods (Access = protected)
%                 
%         function n = maxpartitions(myds)
%             n = maxpartitions(myds.FileSet);
%         end
%     end

end % end class definition