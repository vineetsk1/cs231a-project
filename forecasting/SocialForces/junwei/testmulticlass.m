% Test Multi Class implementation %

TestSet = 3;

Class = clustenergy(TestSet);
close all

[D,T,Obj] = importData();
%save dataset;

% Create tables:
%   Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Obst(dataset,px,py)
%   Dest(dataset,px,py)


[Obsv, Obst, Dest] = data2table(D);
addpath('libsvm-mat-3.0-1/');


%% Global config
Ninterval = 4;
Npast = 8;     % Past steps to use in destination/group prediction

nduration=12;
Nduration = nduration;
fprintf('=== Simulation Config ===\n');
fprintf('  Ninterval: %d\n', Ninterval);
fprintf('  Nduration: %d\n', Nduration);
params = {...
    [1.034086 1.068818 0.822700 0.940671 1.005634 0.842588],...
    [0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420]};
params{1}=[1 1 1 1 1 1 1 1];

time = unique(Obsv(Obsv(:,1)==TestSet,2));
[Obs, Sims] = obsv2sim(Obsv(Obsv(:,1)==TestSet&Obsv(:,2)<=median(time),:),... % Test dataset
    'Interval',Ninterval,...    % Sampling rate
    'Duration',Nduration,...    % Simulation duration
    'Past',Npast);

Cd = [];
Cg = [];


maxDist = 6.0;  % Maximum distance to consider grouping
dt = 0.4;       % Timestep
D = cell(size(Sims,1),1);
for i = 1:size(Sims,1)
    D{i} = Obs(Obs(:,1)==Sims(i,1),:);
end

%% Test

% nargout in Matlab R2009a mac doesn't work with parfor


% Convert Obs and Sims to cell array format
simulations = unique(Sims(:,1));
D = cell(length(simulations),1);
for i = 1:length(simulations)
    D{i} = Obs(Obs(:,1)==simulations(i),:);
end
S = cell(length(simulations),1);
for i = 1:length(simulations)
    S{i} = Sims(Sims(:,1)==simulations(i),:);
end

% Initialization
err = cell(length(simulations),1);
Pred = cell(length(simulations),1);
Grp = cell(length(simulations),1);

% Do simulation
size(simulations)
for i = 1:size(simulations,1)
    
    
    s = Sims(i,:);
    time = unique(D{i}(:,2))';
    time = time(time>=s(4));
    % Set temporary variables
    theta = []; pdest = []; o = [];
    pid = s(3)
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
    theta=params;
    pDest = Dest; pdest = pDest(pDest(:,1)==s(2),2:3);  % dest(x,y)
    pObst = Obst; o = pObst(pObst(:,1)==s(2),2:3);
    cl = Class{TestSet}(pid)
    
    
    for j = 1:length(time)-1
        % Predict dest
        % Output: dhat(j+1)
        if ~isempty(Cd)
            X = D{i}(D{i}(:,2)>=time(max(1,j-Npast))&...
                D{i}(:,2)<=time(j)&...
                D{i}(:,3)==pid,4:7);
            F = destFeature(X(:,1),X(:,2),X(:,3),X(:,4),...
                'XBins',Cd{s(2)}.XBins,...
                'YBins',Cd{s(2)}.YBins,...
                'RhoBins',Cd{s(2)}.RBins,...
                'ThetaBins',Cd{s(2)}.TBins);
            dhat(j+1) = svmpredict(dest(j),F,Cd{s(2)}.C);
        else
            dhat(j+1) = dest(j+1);
        end
        
        % Predict group
        % Output: g{j}
        X = D{i}(D{i}(:,2)==time(j),:);
        others = X(X(:,3)~=pid,3);
        %         others(arrayfun(@(qid)...
        %             sqrt(sum((X(X(:,3)==pid,4:5)-X(X(:,3)==qid,4:5)).^2,2))>maxDist,...
        %             others)) = [];  % Drop too distant guys
        if isempty(others)
            g{j} = zeros(0,5);
        else
            % True label
            Y = arrayfun(@(x) X(X(:,3)==x,10)==X(X(:,3)==pid,10),...
                others);
            if isempty(Cg) % Use grand truth
                Yhat = Y;
            else           % Predict group label
                %
                % TODO: only consider close persons
                %
                X = D{i}(D{i}(:,2)>=time(max(1,j-Npast))&...
                    D{i}(:,2)<=time(j),:);
                F = zeros(length(others),grpFeature('len'));
                for k = 1:length(others)
                    F(k,:) = grpFeature(X(X(:,3)==pid,[2 4:7]),...
                        X(X(:,3)==others(k),[2 4:7]));
                end
                Yhat = logical(svmpredict(double(Y),F,Cg.C));
            end
            g{j} = [repmat(i,[length(Y) 1])...       % simulation id
                repmat(time(j),[length(Y) 1])... % timestamp
                others...                        % other person id
                Y Yhat];                         % true and prediction
        end
        
        
        ind = D{i}(:,2)==time(j)&D{i}(:,3)~=pid; % others index
        groups = logical(arrayfun(@(x) g{j}(g{j}(:,3)==x,5),...
            D{i}(ind,3)));
        vhat(j+1,:) = ...
            fminunc(@(x) myEnergyMultiClass(x,...
            [phat(j,:);o;D{i}(ind,4:5)],... % position
            [vhat(j,:);zeros(size(o));D{i}(ind,6:7)],... % speed
            u(j+1),...                          % ui
            pdest(dhat(j+1),:),...              % zi
            theta,...                           % params
            [true;false(size(o,1),1);groups],... % group indicator
            cl),...                          % class
            vhat(j,:),...                           % init value
            optimset('GradObj','on',...
            'LargeScale','off',...
            'Display','off'...
            ));
    end
    
end
