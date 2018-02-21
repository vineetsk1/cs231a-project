clear all 
close all

load Edetailed


TestSet = [3];
Ttrain = [3];
Ttest = [3];

n = 0;
indtrain=[];indtest=[];
for Ts = TestSet
    
    
    Nped = length(Edetailed{Ts});
    Obsvsel = Obsv(Obsv(:,1)==Ts,:);
    
    for k=1:Nped
        l = length(Edetailed{Ts}{k});
        
        Ops = Obsvsel(Obsvsel(:,3)==k,:);
        
        Ener{k} = struct('damping',zeros(1,l),'destination',zeros(1,l),'attraction',zeros(1,l),...
            'flow',zeros(1,l),'grouping',zeros(1,l),'collision',zeros(1,l),'tot',zeros(1,l));
        for i=1:l
            Ener{n+k}.damping(i) = Edetailed{Ts}{k}(i).damping;
            Ener{n+k}.grouping(i) = Edetailed{Ts}{k}(i).grouping;
            Ener{n+k}.destination(i) = Edetailed{Ts}{k}(i).destination;
            Ener{n+k}.attraction(i) = Edetailed{Ts}{k}(i).attraction;
            Ener{n+k}.flow(i) = Edetailed{Ts}{k}(i).flow;
            Ener{n+k}.tot(i) = Edetailed{Ts}{k}(i).tot;
            Ener{n+k}.collision(i) = Edetailed{Ts}{k}(i).collision;
        end
        
        % Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
        
        
        
        Ep{n+k}(1,:) = Ener{k}.damping;
        Ep{n+k}(2,:) = Ener{k}.grouping;
        Ep{n+k}(3,:) = Ener{k}.destination;
        Ep{n+k}(4,:) = Ener{k}.flow;
        Ep{n+k}(5,:) = Ener{k}.collision;
        Ep{n+k}(6,:) = Ener{k}.attraction;
        Ep{n+k}(7,:) = Ener{k}.tot;
        
        L = size(Ep{n+k},1);
        
        EnAn(n+k,1,1:L+1) = [mean(Ep{n+k},2);Ts];
        EnAn(n+k,2,1:L+1) = [median(Ep{n+k},2);Ts];
        EnAn(n+k,3,1:L+1) = [var(Ep{n+k},0,2);Ts];
        EnAn(n+k,4,1:L+1) = [skewness(Ep{n+k},0,2);Ts];
        EnAn(n+k,5,1:L+1) = [std(Ep{n+k},0,2);Ts];
        EnAn(n+k,6,1:L+1) = [kurtosis(Ep{n+k},0,2);Ts];
        EnAn(n+k,7,1:L+1) = [quantile(Ep{n+k},.9,2);Ts];
        EnAn(n+k,8,1:L+1) = [quantile(Ep{n+k},.8,2);Ts];
        
        if ~isempty(Ops)
            Pedparam{Ts}(k,:) = [max(Ops(:,2))-min(Ops(:,2)) mean(Ops(:,3))...
                mean(Ops(:,4)) median(Ops(:,3)) median(Ops(:,4)) quantile(Ops(:,4),.9)...
                quantile(Ops(:,4),.9) var(Ops(:,3)) var(Ops(:,4)) std(Ops(:,3)) ...
                std(Ops(:,4)) EnAn(n+k,1:8,1)];
        end
        
        for j=1:size(EnAn,3)
            Enfinal{j} = EnAn(:,:,j);
        end
        
    end
    
    n = n+length(Edetailed{Ts});
    
    indtrain = [indtrain; ~isempty(find(Ttrain==Ts))*ones(length(Edetailed{Ts}),1)];
    indtest = [indtest; ~isempty(find(Ttest==Ts))*ones(length(Edetailed{Ts}),1)];
end

close all
a={'damping';'grouping';'destination';'flow';'collision';'attraction'};

for k=1:6
    
    [idx,ctrs] = kmeans(EnAn(:,7:8,k),2);
    X=EnAn(:,7:8,k);
    subplot(2,3,k)
    plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
    hold on
    plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
    plot(ctrs(:,1),ctrs(:,2),'kx',...
        'MarkerSize',12,'LineWidth',2)
    plot(ctrs(:,1),ctrs(:,2),'ko',...
        'MarkerSize',12,'LineWidth',2)
    legend('Cluster 1','Cluster 2','Centroids',...
        'Location','NW')
    title(a{k})
    hold off
    
end

PrNan = length(EnAn(isnan(idx)))/length(EnAn);
nNan = length(EnAn(isnan(idx)));

fprintf('== Number Error NaN %d == \n == Percent Error NaN %d == \n  ',nNan,PrNan);


%% Classification Data

Ptrain = []; Ptest= [];
idxtest=idx(indtest==1); idxtrain=idx(indtrain==1);

for i=Ttrain
    Ptrain = [Ptrain; Pedparam{i}];
end

for i = Ttest
    Ptest = [Ptest; Pedparam{i} ];
end

Ptest = Ptest(isnan(idxtest)==0,:);
Ptrain = Ptrain(isnan(idxtrain)==0,:);
Plabeltrain = idxtrain(isnan(idxtrain)==0,:);
Plabeltest = idxtest(isnan(idxtest)==0,:);

%% Training Classifier

% random forest
b = TreeBagger(40,Ptrain,Plabeltrain,'oobpred','on');
pred = predict(b,Ptrain);
% SVM
SVMStruct = svmtrain(Ptrain,Plabeltrain);


%% detect the optimal number of tree to work with

figure
plot(oobError(b))
xlabel('number of grown trees')
ylabel('out-of-bag classification error')

%%