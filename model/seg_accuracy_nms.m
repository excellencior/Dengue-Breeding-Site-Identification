dataGT = importdata('/home/apurbo/Thesis/Git Repo/final/resources/ClassifierOutput/Building Classification/Ground Truth/labels_binclass.txt',' ');
classGT = dataGT(:,1);
segGT = cat(3,dataGT(:,2:2:9),dataGT(:,3:2:9));
centreGT = reshape(mean(segGT,2),[],2);

numModelData = 6;
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
        case 4
            algo = 'Ground-Truth-Buildings';
        case 5
            algo = 'YOLO-v8m-nms';
        case 6
            algo = 'YOLO-v11m-nms';
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

% Save classification results for each algorithm
for l = 1:numModelData
    algo = result(l).algo;
    dist = result(l).dist;
    
    % Compute final classifications based on the last radius
    finalRadius = result(l).R(end);
    classGT_ = zeros(length(classGT), 1);
    for j = 1:length(classGT)
        if any(dist(result(l).class == 1, j) <= finalRadius)
            classGT_(j) = 1;
        end
    end
    
    % Create output directory if it doesn't exist
    outputDir = fullfile('/home/apurbo/Thesis/Git Repo/final/resources/ClassifierOutput/Building Classification', algo);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    % Create output filename
    outputFile = fullfile(outputDir, 'building_classification.txt');
    
    % Open file for writing
    fid = fopen(outputFile, 'w');
    
    % Write header comment
    fprintf(fid, '# Format: class_id x1 y1 x2 y2 x3 y3 x4 y4\n');
    fprintf(fid, '# class_id: 0 = safe building, 1 = risky building\n');
    
    % Write data for each building
    for i = 1:length(classGT_)
        % Get coordinates from ground truth (since classifications are aligned with GT)
        x_coords = segGT(i,:,1);
        y_coords = segGT(i,:,2);
        
        % Write to file: classification result followed by ground truth coordinates
        fprintf(fid, '%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n', ...
            classGT_(i), ...
            x_coords(1), y_coords(1), ...
            x_coords(2), y_coords(2), ...
            x_coords(3), y_coords(3), ...
            x_coords(4), y_coords(4));
    end
    
    % Close file
    fclose(fid);
    
    fprintf('Saved classification results for %s to: %s\n', algo, outputFile);
end

% % First figure: PR and ROC curves
% figure('OuterPosition',get(groot,'Screensize'));
% colours = 'rgbckm';  % Added 'm' for the sixth model
% linestyles = {'-', '-', '-', '-', ':', ':'};  % Added ':' for the sixth model
% tiledlayout(1,2);

% nexttile;
% for l = 1:numModelData  
%     algo = result(l).algo;
%     TP = result(l).TP;
%     FP = result(l).FP;
%     TN = sum(classGT == 0) - FP;
%     FN = sum(classGT == 1) - TP;
%     precision = [1;TP./(TP + FP);0];
%     recall = [0;TP./(TP + FN);1];
%     AUC = trapz(recall,precision);
%     plot(recall,precision,'.-','Color',colours(l),...
%         'LineStyle',linestyles{l},...
%         'DisplayName',sprintf('%s (AUC $= %0.4f$)',algo,AUC));
%     if l == 1
%         hold on;
%     end
% end
% legend({},'Location','southeast','Interpreter','latex');
% axis equal;
% grid on;
% xlim([0,1]);
% ylim([0,1]);
% xlabel('Recall','Interpreter','latex');
% ylabel('Precision','Interpreter','latex');
% title('PR Curve','Interpreter','latex');
% set(gca,'TickLabelInterpreter','latex','Fontsize',24);
% exportgraphics(gca, 'PR-curve_withIDEAL.png');

% nexttile;
% for l = 1:numModelData  
%     algo = result(l).algo;
%     TP = result(l).TP;
%     FP = result(l).FP;
%     TN = sum(classGT == 0) - FP;
%     FN = sum(classGT == 1) - TP;
%     TPR = [0;TP./(TP + FN);1];
%     FPR = [0;FP./(TN + FP);1];
%     AUC = trapz(FPR,TPR);
%     plot(FPR,TPR,'.-','Color',colours(l),...
%         'LineStyle',linestyles{l},...
%         'DisplayName',sprintf('%s (AUC $= %0.4f$)',algo,AUC));
%     if l == 1
%         hold on;
%     end
% end
% legend({},'Location','southeast','Interpreter','latex');
% axis equal;
% grid on;
% xlim([0,1]);
% ylim([0,1]);
% xlabel('FPR','Interpreter','latex');
% ylabel('TPR','Interpreter','latex');
% title('ROC Curve','Interpreter','latex');
% set(gca,'TickLabelInterpreter','latex','Fontsize',24);
% exportgraphics(gca,'ROC-curve_withIDEAL.png');

% % Second figure: Accuracy and Balanced Accuracy
% figure('OuterPosition',get(groot,'Screensize'));
% colours = 'rbgkcm';  % Added 'm' for the sixth model
% tiledlayout(1,2);

% nexttile;
% l = 2;  % YOLO-v8m
% algo = result(l).algo;
% R = result(l).R;
% TP = result(l).TP;
% FP = result(l).FP;
% TN = sum(classGT == 0) - FP;
% FN = sum(classGT == 1) - TP;
% ACC = (TP + TN)/length(classGT);
% TPR = TP./(TP + FN);
% TNR = TN./(TN + FP);
% bACC = (TPR + TNR)/2;
% plot(R,TPR,'.:','Color',colours(1),'DisplayName',sprintf('TPR-%s',algo));
% hold on;
% plot(R,TNR,'.-.','Color',colours(4),'DisplayName',sprintf('TNR-%s',algo));
% plot(R,ACC,'.-','Color',colours(3),'DisplayName',sprintf('ACC-%s',algo));
% plot(R,bACC,'.--','Color',colours(2),'DisplayName',sprintf('bACC-%s',algo));
% [~,idx] = max(ACC);
% legend({},'Location','northeast','Interpreter','latex');
% grid on;
% xticks([0,200:200:2000]);
% ylim([0,1]);
% xlabel('$r$','Interpreter','latex');
% ylabel('Accuracy','Interpreter','latex');
% title('Accuracy Metrics','Interpreter','latex');
% set(gca,'TickLabelInterpreter','latex','Fontsize',24);
% exportgraphics(gca,'Accuracy_withIDEAL.png');

% nexttile;
% ticks_ = nan(1,numModelData);  
% for l = 1:numModelData  
%     algo = result(l).algo;
%     R = result(l).R;
%     TP = result(l).TP;
%     FP = result(l).FP;
%     TN = sum(classGT == 0) - FP;
%     FN = sum(classGT == 1) - TP;
%     TPR = TP./(TP + FN);
%     TNR = TN./(TN + FP);
%     bACC = (TPR + TNR)/2;
%     plot(R,bACC,'.-','Color',colours(l),...
%         'LineStyle',linestyles{l},...
%         'DisplayName',algo);
%     if l == 1
%         hold on;
%     end
%     [~,idx] = max(bACC);
%     ticks_(l) = R(idx);
%     h = plot(R(idx),bACC(idx),'*','Color',colours(l));
%     set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
%     h = plot([R(idx),R(idx)],[0.5,bACC(idx)],'Color',colours(l),...
%         'LineStyle',linestyles{l});
%     set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% end
% legend({},'Location','northeast','Interpreter','latex');
% grid on;
% ylim([0.5,0.855]);
% xlabel('$r$','Interpreter','latex');
% ylabel('bACC','Interpreter','latex');
% xticks(unique([[0,200:200:2000],max(ticks_)]));
% title('Balanced Accuracy','Interpreter','latex');
% set(gca,'TickLabelInterpreter','latex','Fontsize',24);
% exportgraphics(gca,'Balanced-Accuracy_withIDEAL.png');

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