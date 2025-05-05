dataGT = importdata('/home/apurbo/Thesis/Git Repo/final/resources/ClassifierOutput/Building Classification/Ground Truth/labels_binclass.txt',' ');
classGT = dataGT(:,1);
segGT = cat(3,dataGT(:,2:2:9),dataGT(:,3:2:9));
centreGT = reshape(mean(segGT,2),[],2);

numModelData = 3;
n = 10000;
radii = (0:n)';
result = [];
for l = 1:numModelData  
    switch l
        case 1
            algo = 'SegGPT';
        case 2
            algo = 'YOLO-v8m';
        case 3
            algo = 'YOLO-v11m';
    end
    dataAlgo = ['/home/apurbo/Thesis/Git Repo/final/resources/ClassifierOutput/Building Classification/',algo,'/labels_binclass.txt'];
    disp(dataAlgo)
    dataSeg = importdata(dataAlgo,' ');
    classSeg = dataSeg(:,1);
    segSeg = cat(3,dataSeg(:,2:2:9),dataSeg(:,3:2:9));
    centreSeg = reshape(mean(segSeg,2),[],2);

    closestpoint = nan(length(classSeg),length(classGT),2);
    dist = nan(length(classSeg),length(classGT));
    for i = 1:length(classSeg)
        polySeg = polyshape(segSeg(i,:,1),segSeg(i,:,2));
        for j = 1:length(classGT)
            polyGT = polyshape(segGT(j,:,1),segGT(j,:,2));
            if polyGT.isinterior(centreSeg(i,1),centreSeg(i,2))
                closestpoint(i,j,:) = [centreSeg(i,1),centreSeg(i,2)];
                dist(i,j) = 0;
            else
                closestpoint(i,j,:) = ClosestPoint(reshape(segGT(j,:,:),[],2),[centreSeg(i,1),centreSeg(i,2)]);
                dist(i,j) = norm(reshape(closestpoint(i,j,:),1,[]) - centreSeg(i,:));
            end
        end
    end

    TP = nan(n+1, 1);
    FP = nan(n+1, 1);
    for k = 0:n
        classGT_ = zeros(length(classGT),1);
        for j = 1:length(classGT)
            if any(dist(classSeg == 1,j) <= radii(k + 1))
                classGT_(j) = 1;
            end
        end
        TP(k + 1) = sum(classGT_ == 1 & classGT == 1);
        FP(k + 1) = sum(classGT_ == 1 & classGT == 0);
        if all(classGT_ == 1)
            break
        end
    end
    data = [TP(1:k+1),FP(1:k+1),radii(1:k+1)];
    [~,ia] = unique(data(:,1:2),'stable','rows');
    TP = data(ia,1);
    FP = data(ia,2);
    R = data(ia,3);
    data = struct(...
        'algo',algo,...
        'seg',segSeg,...
        'class',classSeg,...
        'closestpoint',closestpoint,...
        'dist',dist,...
        'TP',TP,...
        'FP',FP,...
        'R',R);
    result = [result;data];
end

% Save classification results for YOLO-v8m
for l = 1:numModelData
    algo = result(l).algo;
    
    % Only process YOLO-v8m
    if ~strcmp(algo, 'YOLO-v8m')
        continue;
    end
    
    % Setup for classification with r = 83
    radius = 83;
    dist = result(l).dist;
    distMax = max(dist(:));  % Get maximum distance for normalization
    th = radius / distMax;   % Threshold for normalized distance
    
    % Find risky buildings (class_id = 1)
    risky_buildings = find(any(dist(result(l).class == 1, :) / distMax <= th));
    
    % Create output directory if it doesn't exist
    outputDir = fullfile('/home/apurbo/Thesis/Git Repo/final/resources/ClassifierOutput/Building Classification', algo);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    % Create output filename
    outputFile = fullfile(outputDir, 'building_classification.txt');
    
    % Open file for writing
    fileID = fopen(outputFile, 'w');
    
    % Write data for all buildings (total 297)
    for i = 1:length(classGT)
        % Determine class_id (1 for risky, 0 for safe)
        class_id = ismember(i, risky_buildings);
        
        % Get the coordinates of the 4 corners
        corners = segGT(i, :, :);
        corners = reshape(corners, [], 2);
        
        % Write class id and coordinates in the required format
        fprintf(fileID, '%d %d %d %d %d %d %d %d %d\n', ...
            class_id, ...
            round(corners(1,1)), round(corners(1,2)), ...
            round(corners(2,1)), round(corners(2,2)), ...
            round(corners(3,1)), round(corners(3,2)), ...
            round(corners(4,1)), round(corners(4,2)));
    end
    
    % Close file
    fclose(fileID);
    
    fprintf('Saved classification results for %s to: %s\n', algo, outputFile);
    fprintf('Total buildings processed: %d\n', length(classGT));
end

function P_ = ClosestPoint(seg,P)
    poly = polyshape(seg(:,1),seg(:,2));
    P_ = nan(4,2);
    for i = 1:4
        P_(i,:) = Foot(poly.Vertices(i,:)',poly.Vertices(mod(i,4) + 1,:)',P')';
    end
    [~,idx] = min(sqrt(sum((P - P_).^2,2)));
    P_ = P_(idx,:);

    function P_ = Foot(A,B,P)
        P_ = A + (B - A)\(P - A)*(B - A);
        AB = norm(A - B);
        AP_ = norm(A - P_);
        BP_ = norm(B - P_);
        if AP_ > AB || BP_ > AB
            if AP_ < BP_
                P_ = A;
            else
                P_ = B;
            end
        end
    end
end