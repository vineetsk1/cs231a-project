function class = clustenergy(TestSet)

load Edetailed
load Pedfeat

n = 0;
for Ts = TestSet
    
    
    Nped = length(Edetailed{Ts});
    %Obsvsel = Obsv(Obsv(:,1)==Ts,:);
    
    for k=1:Nped
        l = length(Edetailed{Ts}{k});
        
        %Ops = Obsvsel(Obsvsel(:,3)==k,:);
        PedFeat{n+k} = struct('px',zeros(1,l),'py',zeros(1,l),'set',zeros(1,l),...
            'vx',zeros(1,l),'vy',zeros(1,l),'speed',zeros(1,l),'group',zeros(1,l));
        Ener{n+k} = struct('damping',zeros(1,l),'destination',zeros(1,l),'attraction',zeros(1,l),...
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
        
        
        K = [4 5 6 7 9 10];
        L1 = length(K);
        % Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
        
        PF(n+k,1:6*L1) = [mean(Pedfeat{Ts}{k}(:,K)) ...
            median(Pedfeat{Ts}{k}(:,K)) ...
            var(Pedfeat{Ts}{k}(:,K)) ...
            kurtosis(Pedfeat{Ts}{k}(:,K)) ...
            quantile(Pedfeat{Ts}{k}(:,K),.8) ...
            quantile(Pedfeat{Ts}{k}(:,K),.1)];
            %skewness(Pedfeat{Ts}{k}(:,K)) ...
            %std(Pedfeat{Ts}{k}(:,K)) ...
            
        
        
        
        Ep{n+k}(1,:) = Ener{k}.damping;
        Ep{n+k}(2,:) = Ener{k}.grouping;
        Ep{n+k}(3,:) = Ener{k}.destination;
        Ep{n+k}(4,:) = Ener{k}.flow;
        Ep{n+k}(5,:) = Ener{k}.collision;
        Ep{n+k}(6,:) = Ener{k}.attraction;
        Ep{n+k}(7,:) = Ener{k}.tot;
        
        L = size(Ep{n+k},1);%
        
        EnAn(n+k,1,1:L+1) = [mean(Ep{n+k},2);Ts];
        EnAn(n+k,2,1:L+1) = [median(Ep{n+k},2);Ts];
        EnAn(n+k,3,1:L+1) = [var(Ep{n+k},0,2);Ts];
        EnAn(n+k,4,1:L+1) = [skewness(Ep{n+k},0,2);Ts];
        EnAn(n+k,5,1:L+1) = [std(Ep{n+k},0,2);Ts];
        EnAn(n+k,6,1:L+1) = [kurtosis(Ep{n+k},0,2);Ts];
        EnAn(n+k,7,1:L+1) = [quantile(Ep{n+k},.9,2);Ts];
        EnAn(n+k,8,1:L+1) = [quantile(Ep{n+k},.8,2);Ts];
        
        for j=1:size(EnAn,3)
            Enfinal{j} = EnAn(:,:,j);
        end
        
    end
    
    n = n+length(Edetailed{Ts});
end


indk=[];
for k=1:size(PF,2)
    if sum(isnan(PF(:,k)))>0
        indk = [indk k];
    end
end


PF(:,indk) = [];
Z = linkage(PF);

close all
a={'damping';'grouping';'destination';'flow';'collision';'attraction'};

j=0;
for k=[2 3 5 6]
    j=j+1;
    [idx,ctrs] = kmeans([EnAn(:,7:8,k)],2,'start','cluster');
    X=EnAn(:,7:8,k);
    subplot(2,2,j)
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
    class{k} = idx;
    
end



PrNan = length(EnAn(isnan(idx)))/length(EnAn);
nNan = length(EnAn(isnan(idx)));

%fprintf('== Number Error NaN %d == \n == Percent Error NaN %d == \n  ',nNan,PrNan);
end

