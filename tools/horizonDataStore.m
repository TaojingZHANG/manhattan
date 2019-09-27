classdef horizonDataStore < matlab.io.Datastore & ...
                       matlab.io.datastore.MiniBatchable & ...
                       matlab.io.datastore.Shuffleable & ...
                       matlab.io.datastore.PartitionableByIndex
    
    properties
        Datastore
        Labels
        MiniBatchSize
        imSize
        meanImage
        bw % use black and white?
    end
    
    properties(SetAccess = protected)
        NumObservations
    end

    properties(Access = private)
        % This property is inherited from Datastore
        CurrentFileIndex
    end


    methods
        
        function ds = horizonDataStore(imDs, inputLabels, imSize, meanImage, bw)

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
            ds.meanImage = meanImage;
            ds.bw = bw;
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
                origSize = size(X);
                if origSize(1) > origSize(2) % vertical
                  squareSize = origSize(2);
                  L = origSize(1);
                  interval = [ceil(L / 2) - floor(squareSize / 2):ceil(L / 2) + floor(squareSize / 2) - 1];
                  Xcrop = X(interval, :, :);
                elseif origSize(2) > origSize(1) % horizontal
                  squareSize = origSize(1);
                  L = origSize(2);
                  interval = [ceil(L / 2) - floor(squareSize / 2):ceil(L / 2) + floor(squareSize / 2) - 1];
                  Xcrop = X(:, interval, :);
                else % square
                  Xcrop = X;
                end
                
                Xsmall = imresize(Xcrop, ds.imSize);
                                
                if ds.bw
                  predictors{i} = single(rgb2gray(Xsmall)) - single(ds.meanImage);
                else
                  predictors{i} = single(Xsmall) - single(ds.meanImage);
                end
                predictors{i} = predictors{i} / 127; % normalize to [-1, 1]
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