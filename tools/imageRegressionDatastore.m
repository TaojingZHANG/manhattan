classdef imageRegressionDatastore < matlab.io.Datastore & ...
                       matlab.io.datastore.MiniBatchable & ...
                       matlab.io.datastore.Shuffleable & ...
                       matlab.io.datastore.PartitionableByIndex
                       %matlab.io.datastore.Partitionable & ...
    
    properties
        Datastore
        Labels
        Intrinsics
        MiniBatchSize
    end
    
    properties(SetAccess = protected)
        NumObservations
    end

    properties(Access = private)
        % This property is inherited from Datastore
        CurrentFileIndex
    end


    methods
        
        function ds = imageRegressionDatastore(folder, inputLabels, intrinsics)

            % Create a file datastore. The readSequence function is
            % defined following the class definition.
            fds = fileDatastore(folder, ...
                'ReadFcn',@imread, ...
                'IncludeSubfolders',true);
            ds.Datastore = fds;

            % Read labels from folder names
            numObservations = numel(fds.Files);
            for i = 1:numObservations
                file = fds.Files{i};
                filepath = fileparts(file);
                %[~,label] = fileparts(filepath);
                labels{i,1} = inputLabels(1, 1, :, i);
            end
            ds.Labels = labels;
            ds.Intrinsics = intrinsics;
            
            % Initialize datastore properties.
            ds.MiniBatchSize = 128;
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;
        end
        
%         function subds = partition(myds,n,ii)
%             subds = copy(myds);
%             subds.FileSet = partition(myds.FileSet,n,ii);
%             reset(subds);
%         end
        
        function subds = partitionByIndex(myds, indices)
            subds = copy(myds);
            subds.Datastore = copy(myds.Datastore);
            subds.Datastore.Files = myds.Datastore.Files(indices);
            subds.Labels = myds.Labels(indices);
            for i = 1:length(myds.Intrinsics)
                subds.Intrinsics{i} = myds.Intrinsics{i}(indices);
            end
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
            
            % First normalize the camera then
            % perform edge detection and downsample
            for i = 1:miniBatchSize
                X = predictors{i};
                imSize=size(X);
                
                X = correctDistortion(ds, X, imSize(1:2), fileIndices(i));
                
                %predictors{i} = edge(rgb2gray(imresize(X, 1 / 4)), 'canny');
                predictors{i} = rgb2gray(imresize(X, 1 / 4));
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
            
            for i = 1:length(dsNew.Intrinsics)
                dsNew.Intrinsics{i} = dsNew.Intrinsics{i}(idx);
            end 
        end
        
        function Xcal = correctDistortion(ds, X, imSize, fileIndex)
%           w = ds.Intrinsics{2}(fileIndex);
%           h = ds.Intrinsics{3}(fileIndex);
          f = ds.Intrinsics{4}(fileIndex);
          cx = ds.Intrinsics{5}(fileIndex);
          cy = ds.Intrinsics{6}(fileIndex);
          r = ds.Intrinsics{7}(fileIndex);
          
          camParams = cameraIntrinsics(f, [cx, cy], imSize, ...
            'RadialDistortion', [r, 0]);
          
          Xcal = undistortImage(X, camParams);
          
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