dataGT = importdata('/home/apurbo/Thesis/Git Repo/final/resources/ClassifierOutput/Building Classification/Ground Truth/labels_binclass.txt',' ');
% dataGT = importdata('/home/apurbo/Thesis/Git Repo/final/Labels/Building Classification/Ground Truth/labels_binclass.txt',' ');
classGT = dataGT(:,1);
segGT = cat(3,dataGT(:,2:2:9),dataGT(:,3:2:9));
centreGT = reshape(mean(segGT,2),[],2);

n = 10000;
radii = (0:n)';
result = [];
for l = 1:3
    switch l
        case 1
            algo = 'SegGPT';
        case 2
            algo = 'YOLO-v8m';
        otherwise
            algo = 'YOLO-v11m';
    end
    dataSeg = importdata(['/home/apurbo/Thesis/Git Repo/final/resources/ClassifierOutput/Building Classification/',algo,'/labels_binclass.txt'],' ');
    % dataSeg = importdata(['/home/apurbo/Thesis/Git Repo/final/Labels/Building Classification/',algo,'/labels_binclass.txt'],' ');
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

    % TP = nan(1, n+1);
    % FP = nan(1, n+1);
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

% Needs result structure beyond this point

figure('OuterPosition',get(groot,'Screensize'));
colours = 'rgb';
tiledlayout(1,2);
nexttile;
for l = 1:3
    algo = result(l).algo;
    TP = result(l).TP;
    FP = result(l).FP;
    TN = sum(classGT == 0) - FP;
    FN = sum(classGT == 1) - TP;
    precision = [1;TP./(TP + FP);0];
    recall = [0;TP./(TP + FN);1];
    AUC = trapz(recall,precision);
    plot(recall,precision,'.-','Color',colours(l),...
        'DisplayName',sprintf('%s (AUC $= %0.4f$)',algo,AUC));
    if l == 1
        hold on;
    end
end
legend({},'Location','southeast','Interpreter','latex');
axis equal;
grid on;
xlim([0,1]);
ylim([0,1]);
xlabel('Recall','Interpreter','latex');
ylabel('Precision','Interpreter','latex');
title('PR Curve','Interpreter','latex');
set(gca,'TickLabelInterpreter','latex','Fontsize',24);
% exportgraphics(gca, 'PR-curve.png');
exportgraphics(gca,'PR-curve.emf');
nexttile;
for l = 1:3
    algo = result(l).algo;
    TP = result(l).TP;
    FP = result(l).FP;
    TN = sum(classGT == 0) - FP;
    FN = sum(classGT == 1) - TP;
    TPR = [0;TP./(TP + FN);1];
    FPR = [0;FP./(TN + FP);1];
    AUC = trapz(FPR,TPR);
    plot(FPR,TPR,'.-','Color',colours(l),...
        'DisplayName',sprintf('%s (AUC $= %0.4f$)',algo,AUC));
    if l == 1
        hold on;
    end
end
legend({},'Location','southeast','Interpreter','latex');
axis equal;
grid on;
xlim([0,1]);
ylim([0,1]);
xlabel('FPR','Interpreter','latex');
ylabel('TPR','Interpreter','latex');
title('ROC Curve','Interpreter','latex');
set(gca,'TickLabelInterpreter','latex','Fontsize',24);
% exportgraphics(gca, 'ROC-curve.png')
exportgraphics(gca,'ROC-curve.emf');

figure('OuterPosition',get(groot,'Screensize'));
colours = 'rbgk';
tiledlayout(1,2);
nexttile;
l = 2;
algo = result(l).algo;
R = result(l).R;
TP = result(l).TP;
FP = result(l).FP;
TN = sum(classGT == 0) - FP;
FN = sum(classGT == 1) - TP;
ACC = (TP + TN)/length(classGT);
TPR = TP./(TP + FN);
TNR = TN./(TN + FP);
bACC = (TPR + TNR)/2;
plot(R,TPR,'.:','Color',colours(1),'DisplayName',sprintf('TPR-%s',algo));
hold on;
plot(R,TNR,'.-.','Color',colours(4),'DisplayName',sprintf('TNR-%s',algo));
plot(R,ACC,'.-','Color',colours(3),'DisplayName',sprintf('ACC-%s',algo));
plot(R,bACC,'.--','Color',colours(2),'DisplayName',sprintf('bACC-%s',algo));
[~,idx] = max(ACC);
legend({},'Location','northeast','Interpreter','latex');
grid on;
xticks([0,200:200:2000]);
ylim([0,1]);
xlabel('$r$','Interpreter','latex');
ylabel('Accuracy','Interpreter','latex');
title('Accuracy Metrics','Interpreter','latex');
set(gca,'TickLabelInterpreter','latex','Fontsize',24);
% exportgraphics(gca, 'Accuracy.png')
exportgraphics(gca,'Accuracy.emf');
nexttile;
ticks_ = nan(1,3);
for l = 1:3
    algo = result(l).algo;
    R = result(l).R;
    TP = result(l).TP;
    FP = result(l).FP;
    TN = sum(classGT == 0) - FP;
    FN = sum(classGT == 1) - TP;
    TPR = TP./(TP + FN);
    TNR = TN./(TN + FP);
    bACC = (TPR + TNR)/2;
    plot(R,bACC,'.-','Color',colours(l),'DisplayName',algo);
    if l == 1
        hold on;
    end
    [~,idx] = max(bACC);
    ticks_(l) = R(idx);
    h = plot(R(idx),bACC(idx),'*','Color',colours(l));
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    h = plot([R(idx),R(idx)],[0.5,bACC(idx)],'Color',colours(l));
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end
legend({},'Location','northeast','Interpreter','latex');
grid on;
ylim([0.5,0.855]);
xlabel('$r$','Interpreter','latex');
ylabel('bACC','Interpreter','latex');
xticks(unique([[0,200:200:2000],max(ticks_)]));
title('Balanced Accuracy','Interpreter','latex');
set(gca,'TickLabelInterpreter','latex','Fontsize',24);
% exportgraphics(gca, 'Balanced-Accuracy.png')
exportgraphics(gca,'Balanced-Accuracy.emf');
5;

figure('OuterPosition',get(groot,'Screensize'));
jj = [87,245,247,248,249];
radii = [0,425,475,600,800];
tiledlayout(1,length(radii));
for radius = radii
    nexttile;
    i = 163;
    closestpoint = reshape(result(3).closestpoint(i,:,:),[],2);
    dist = result(3).dist(i,:);
    centreSeg = reshape(mean(result(3).seg(i,:,:),2),[],2);
    for j = jj
        if dist(j) <= radius
            colour = 'r';
            linestyle = '--';
        else
            colour = 'k';
            linestyle = '-';
        end
        plot(segGT(j,[1,2,3,4,1],1),segGT(j,[1,2,3,4,1],2),...
            'LineStyle',linestyle,'Color',colour,'LineWidth',1.5);
        if j == jj(1)
            hold on;
        end
        text(centreGT(j,1),centreGT(j,2),sprintf('%d',j),'color','k','FontSize',14,...
            'HorizontalAlignment','center','VerticalAlignment','middle');
    end
    axis equal;
    axis tight;
    xlim_ = xlim;
    ylim_ = ylim;
    rectangle('Position',[centreSeg(1) - radius,centreSeg(2) - radius,radius*2,radius*2],...
        'Curvature',[1,1],'FaceColor',[1,1,0,0.25],'EdgeColor','y','LineWidth',1);
    plot(centreSeg(1),centreSeg(2),'.r');
    text(centreSeg(1),centreSeg(2),sprintf('%d',i),'color','r','FontSize',14,...
        'HorizontalAlignment','center','VerticalAlignment','bottom');
    ids = find(dist(jj) <= radius);
    for j = jj(ids)
        quiver(centreSeg(1),centreSeg(2),...
            closestpoint(j,1) - centreSeg(1),closestpoint(j,2) - centreSeg(2),...
            'off','Color','k');
    end
    xlim(xlim_);
    ylim(ylim_);
    xlabel(sprintf('$r = %d$',radius),'Interpreter','latex')
    if isempty(ids)
        title_ = sprintf('No risky-building detected');
    elseif length(ids) == 1
        title_ = sprintf('%d risky-building detected',length(ids));
    else
        title_ = sprintf('%d risky-buildings detected',length(ids));
    end
    title(title_,'Interpreter','latex');
    grid on;
    set(gca,'TickLabelInterpreter','latex','Fontsize',18);
end
exportgraphics(gcf, 'Building-Detection.png')
exportgraphics(gcf,'building-detection.emf');

figure('OuterPosition',get(groot,'Screensize'));
radii = [0,83];
% radii = [200,400];
tiledlayout(length(radii),3,'TileIndexing','columnmajor');
for l = 1:3
    algo = result(l).algo;
    classSeg = result(l).class;
    segSeg = result(l).seg;
    centreSeg = reshape(mean(segSeg,2),[],2);
    dist = result(l).dist;
    distMax = max(dist(classSeg == 1,:),[],'all');
    closestpoint = result(l).closestpoint;
    for t = 1:length(radii)
        nexttile;
        th = radii(t)/distMax;
        classGT_ = zeros(length(classGT),1);
        for j = 1:length(classGT)
            if any(dist(classSeg == 1,j)/distMax <= th)
                classGT_(j) = 1;
            end
        end
        TP = sum(classGT_ == 1 & classGT == 1);
        FP = sum(classGT_ == 1 & classGT == 0);
        TN = sum(classGT_ == 0 & classGT == 0);
        FN = sum(classGT_ == 0 & classGT == 1);
        precision = TP/(TP + FP);
        recall = TP/(TP + FN);
        TPR = TP/(TP + FN);
        TNR = TN/(TN + FP);
        BalAcc = (TPR + TNR)/2;
        for j = 1:length(classGT)
            if classGT(j) == 0 && classGT_(j) == 0
                colour = 'b';
            elseif classGT(j) == 0 && classGT_(j) == 1
                colour = 'c';
            elseif classGT(j) == 1 && classGT_(j) == 0
                colour = 'm';
            else
                colour = 'r';
            end
            plot(segGT(j,[1,2,3,4,1],1),segGT(j,[1,2,3,4,1],2),'-','Color',colour,'LineWidth',1);
            if j == 1
                hold on;
            end
            % text(centreGT(j,1),centreGT(j,2),sprintf('%d',j),'color',colour,'FontSize',5,...
            %     'HorizontalAlignment','center','VerticalAlignment','middle');
        end
        for i = 1:length(classSeg)
            if classSeg(i) == 1
                radius = th*distMax;
                rectangle('Position',[centreSeg(i,1) - radius,centreSeg(i,2) - radius,radius*2,radius*2],...
                    'Curvature',[1,1],'FaceColor',[1,1,0,0.25],'EdgeColor','y','LineWidth',1);
                plot(centreSeg(i,1),centreSeg(i,2),'.r');
                for j = find(dist(i,:)/distMax <= th)
                    quiver(centreSeg(i,1),centreSeg(i,2),...
                        closestpoint(i,j,1) - centreSeg(i,1),closestpoint(i,j,2) - centreSeg(i,2),...
                        'off','Color','k');
                end
            end
        end
        axis equal;
        axis tight;
        xlim_ = xlim;
        xticks(0:2000:xlim_(2));
        ylim_ = ylim;
        yticks(0:2000:ylim_(2));
        grid on;
        if t == length(radii)
            xlabel(algo,'Interpreter','latex');
        end
        if l == 1
            ylabel(sprintf('$r = %d$',radii(t)),'Interpreter','latex');
        end
        title(sprintf('TP:%d; FP:%d; P-R:%0.3f-%0.3f; bACC:%0.3f',...
            TP,FP,precision,recall,BalAcc),'Interpreter','latex');
        set(gca,'TickLabelInterpreter','latex','Fontsize',14);
        drawnow;
    end
end
% exportgraphics(gcf, 'Building-Classes.png')
exportgraphics(gcf,'Building-classes.emf');


function P_ = ClosestPoint ( seg,P )

poly = polyshape(seg(:,1),seg(:,2));
P_ = nan(4,2);
for i = 1:4
    P_(i,:) = Foot(poly.Vertices(i,:)',poly.Vertices(mod(i,4) + 1,:)',P')';
end
[~,idx] = min(sqrt(sum((P - P_).^2,2)));
P_ = P_(idx,:);

    function P_ = Foot ( A,B,P )

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
