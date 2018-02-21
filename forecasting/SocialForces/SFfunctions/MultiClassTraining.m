function [SVMclass,clust]=MultiClassTraining(D)


% Create tables:
%   Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Obst(dataset,px,py)
%   Dest(dataset,px,py)
[Obsv, Obst, Dest] = data2table(D);
addpath('libsvm-mat-3.0-1/');

%% Global config/ Training


Ninterval = 400; % Interval to subsample simulation sequence
Npast = 8;   % Past steps to include
Cd=[];
Cg=[];

fprintf('=== SVM Config ===\n');
fprintf('  Ninterval: %d\n', Ninterval);
fprintf('  Npast: %d\n', Npast);
Sigtot=[];


%load Cg Cd

Nduration = 1200;  % Future steps to include (= #simulation steps)

params = [0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420];

% Create simulation tables:
%   Obs(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Sims(simulation,dataset,person,start,duration)


for TestSet = 1:6
    Psig{TestSet}=[];Tsig{TestSet}=[];
    time = unique(Obsv(Obsv(:,1)==TestSet,2));
    [Obs, Sims] = obsv2sim(Obsv(Obsv(:,1)==TestSet&Obsv(:,2)<=median(time),:),... % Test dataset
        'Interval',Ninterval,...    % Sampling rate
        'Duration',Nduration,...    % Simulation duration
        'Past',Npast);
    
    
    % Convert Obs to cell array format
    D = cell(size(Sims,1),1);
    for i = 1:size(Sims,1)
        D{i} = Obs(Obs(:,1)==Sims(i,1),:);
    end
    
    %% Ped Characteristic
    
    
    %% Initialization
    Nargout = nargout; % nargout in Matlab R2009a mac doesn't work with parfor
    err = cell(size(Sims,1),1);
    if Nargout > 1
        Pred = cell(size(Sims,1),1);
        Grp = cell(size(Sims,1),1);
    end
    
    % Do simulation
    for i = 1:size(Sims,1)
        % Query simulation info
        s = Sims(i,:);
        time = unique(D{i}(:,2))';
        time = time(time>=s(4));
        % Set temporary variables
        
        theta = params;                                     % parameters
        pDest = Dest; pdest = pDest(pDest(:,1)==s(2),2:3);  % dest(x,y)
        pObst = Obst; o = pObst(pObst(:,1)==s(2),2:3);      % obst(x,y)
        
        %% Initialize variables for simulation
        pid = s(3);
        p = D{i}(D{i}(:,2)>=s(4)&D{i}(:,3)==pid,4:5); % positions
        v = D{i}(D{i}(:,2)==s(4)&D{i}(:,3)==pid,6:7); % velocity
        phat = zeros(size(p));          % list of predicted positions of pid
        phat(1,:) = p(1,:);             % initial position is grand truth
        vhat = repmat(v,[size(p,1) 1]); % list of predicted velocity of pid
        dest = D{i}(D{i}(:,2)>=s(4)&D{i}(:,3)==pid,8); % dest
        dhat = zeros(size(dest));       % dest prediction
        dhat(1) = dest(1);
        u = D{i}(D{i}(:,2)>=s(4)&D{i}(:,3)==pid,9); % speed
        g = cell(length(time)-1,1);
        
        Pedfeat{TestSet}{pid} = D{i};
        sigmarl{TestSet}{pid}=[];
        % group table skelton
        
        %% Simulate the behavior of person pid
        for j = 1:length(time)-1
            %% Predict dest
            dhat(j+1) = dest(j+1);
            
            %% Predict group
            
            X = D{i}(D{i}(:,2)==time(j),:);
            others = X(X(:,3)~=pid,3);
            
            if isempty(others)
                g{j} = zeros(0,5);
            else
                % True label
                try
                    Y = arrayfun(@(x) X(X(:,3)==x,10)==X(X(:,3)==pid,10),...
                        others);
                end
                if isempty(Y)
                    g{j} = zeros(0,5);
                else
                    Yhat = Y;
                    
                    g{j} = [repmat(i,[length(Y) 1])...       % simulation id
                        repmat(time(j),[length(Y) 1])... % timestamp
                        others...                        % other person id
                        Y Yhat];
                end% true and prediction
            end
            
            %% Predict velocity and position
            %fprintf('energy')
            % Output: vhat(j+1,:), phat(j+1,:)
            ind = D{i}(:,2)==time(j)&D{i}(:,3)~=pid; % others index
            groups = logical(arrayfun(@(x) g{j}(g{j}(:,3)==x,5),...
                D{i}(ind,3)));
            
            % we consider the true speed to evaluate the energy
            dt=D{i}(2,2)-D{i}(1,2);
            vhat(j+1,:) = D{i}(D{i}(:,2)==s(4)+j*dt&D{i}(:,3)==pid,6:7);
            
            [e,~] = myEnergyDetailed(vhat(j+1,:),...
                [phat(j,:);o;D{i}(ind,4:5)],... % position
                [vhat(j,:);zeros(size(o));D{i}(ind,6:7)],... % speed
                u(j+1),...                          % ui
                pdest(dhat(j+1),:),...              % zi
                theta,...                           % params
                [true;false(size(o,1),1);groups]);...   % group indicator
                vhat(j,:),...                           % init value
                optimset('GradObj','on',...
                'LargeScale','off',...
                'Display','off'...
                );
            Edetailed{TestSet}{pid}.collision(j)=e.collision;
            Edetailed{TestSet}{pid}.time(j) = time(j);
            Edetailed{TestSet}{pid}.pedestrians{j} = unique(D{i}(:,3));
            
            if Edetailed{TestSet}{pid}.collision(j)>0.15
                [sr,dsig] = fmincon (@(x) Ecollision(x,...
                    [theta(2) theta(3)],...
                    vhat(j+1,:),...
                    [phat(j,:);D{i}(ind,4:5)],... % position
                    [vhat(j,:);D{i}(ind,6:7)]),...
                    theta(1),-1,0.1);
                
                sigmarl{TestSet}{pid} = [sigmarl{TestSet}{pid}; sr  Edetailed{TestSet}{pid}.time(j)];
                Psig{TestSet}=[Psig{TestSet} pid];
                Tsig{TestSet}=[Tsig{TestSet} time(j)];
                
            end
            phat(j+1,:) = phat(j,:) + dt*vhat(j+1,:);  % position
            
        end
    end
    
    Sig{TestSet}=[];
    for i=1:length(sigmarl{TestSet})
        
        if ~isempty(sigmarl{TestSet}{i})
            Sig{TestSet} = [Sig{TestSet};i*ones(length(sigmarl{TestSet}{i}(:,1)),1)...
                sigmarl{TestSet}{i}(:,1) ];
        end
    end
    Sig{TestSet} = [Sig{TestSet} Tsig{TestSet}'];
    Sigtot = [Sigtot; Sig{TestSet}(:,2)];
end

%% Cluster

eva = evalclusters(Sigtot(:,1),'kmeans','CalinskiHarabasz','KList',[1:10]);

[idx,clust]=kmeans(Sigtot(:,1),eva.OptimalK,'start','cluster');

%Sig(pedestrian,sigma,SI


K=1; Xtrain=[];Ytrain=[];
for k=1:5
    Sig{k} = [Sig{k} idx(K:K+length(Sig{k})-1)];
    K = K+length(Sig{k});
    O = Obsv(Obsv(:,1)==k,:);
    for i=1:size(Sig{k},1);
        Xtrain = [Xtrain; O(O(:,2)==Sig{k}(i,3)&O(:,3)==Sig{k}(i,1),:)];
        Ytrain = [Ytrain; k Sig{k}(i,4)];
    end
    SVMclass{k}=svmtrain(Ytrain(Ytrain(:,1)==k,2),Xtrain(Xtrain(:,1)==k,2:end-2));
    svmpredict(Ytrain(Ytrain(:,1)==k,2),Xtrain(Xtrain(:,1)==k,2:end-2),SVMclass{k});
end
end
