function [ err, Pred, Grp ] = simErrorMultiClass( Obs, Sims, method, varargin )
%SIMERROR behavioral simulation and error evaluation
%
% Input:
%   Obs(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Sims(simulation,dataset,person,start,duration)
%   method: either of 'lin', 'ewap', 'attr'
%   options:
% Output:
%   err: error array for each simulation step
%   Pred: result table for each simulation step
%         (simulation,time,person,px,py,vx,vy,dest,speed
%   Grp: predicted group relation for the subject of simulation
%         (simulation,time,pid)
%        That is, pid in the Sims table has relation to pid in the Grp


% Options
Cd = [];
Cg = [];
Npast = 0;      % Past steps to use in destination/group prediction
params = [];
Dest = [];
Obst = [];
Class = [];
maxDist = 6.0;  % Maximum distance to consider grouping
dt = 0.4;       % Timestep
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'DestClassifier'), Cd = varargin{i+1}; end
    if strcmp(varargin{i},'GroupClassifier'), Cg = varargin{i+1}; end
    if strcmp(varargin{i},'Past'), Npast = varargin{i+1}; end
    if strcmp(varargin{i},'Params'), params = varargin{i+1}; end
    if strcmp(varargin{i},'Dest'), Dest = varargin{i+1}; end
    if strcmp(varargin{i},'Obst'), Obst = varargin{i+1}; end
    if strcmp(varargin{i},'Class'), Class = varargin{i+1}; end
    if strcmp(varargin{i},'MaxDist'), maxDist = varargin{i+1}; end
end

%% Check args
if ~any(cellfun(@(x)isempty(x),...
        strfind({'lin','ewap','attr'},method)))
    error('Invalid method');
end
if strcmp(method,'ewap') || strcmp(method,'attr')
    if isempty(params) || isempty(Dest), error('Missing arguments'); end
end

% Convert Obs to cell array format
D = cell(size(Sims,1),1);
for i = 1:size(Sims,1)
    D{i} = Obs(Obs(:,1)==Sims(i,1),:);
end

%% Initialization
Nargout = nargout; % nargout in Matlab R2009a mac doesn't work with parfor
err = cell(size(Sims,1),1);
if Nargout > 1
    Pred = cell(size(Sims,1),1);
    Grp = cell(size(Sims,1),1);
end
% Do simulation
parfor i = 1:size(Sims,1)
    % Query simulation info
    s = Sims(i,:);
    time = unique(D{i}(:,2))';
    time = time(time>=s(4));
    % Set temporary variables
    theta = []; pdest = []; o = [];
    if ~strcmp(method,'lin')
        theta = params;                                     % parameters
        pDest = Dest; pdest = pDest(pDest(:,1)==s(2),2:3);  % dest(x,y)
        pObst = Obst; o = pObst(pObst(:,1)==s(2),2:3);      % obst(x,y)
    end
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
    g = cell(length(time)-1,1);                 % group table skelton
    %% Simulate the behavior of person pid
    for j = 1:length(time)-1
        %% Predict speed?
        % TODO: estimate u
        
        %% Predict dest
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
        
        %% Predict group
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

        %% Predict velocity and position
        % Output: vhat(j+1,:), phat(j+1,:)
        if strcmp(method,'ewap')
            ind = D{i}(:,2)==time(j)&D{i}(:,3)~=pid; % others index
            vhat(j+1,:) = theta(6)* vhat(j,:) + (1-theta(6)) *...
                fminunc(@(x) ewapEnergy(x,...
                    [phat(j,:);o;D{i}(ind,4:5)],... % position
                    [vhat(j,:);zeros(size(o));D{i}(ind,6:7)],... % speed
                    u(j+1),...                      % ui
                    pdest(dhat(j+1),:),...          % zi
                    theta),...                      % params
                vhat(j,:),...                       % init value
                optimset('GradObj','on',...
                         'LargeScale','off',...
                         'Display','off'...
                    ));
        elseif strcmp(method,'attr')
            ind = D{i}(:,2)==time(j)&D{i}(:,3)~=pid; % others index
            groups = logical(arrayfun(@(x) g{j}(g{j}(:,3)==x,5),...
                                      D{i}(ind,3)));
            vhat(j+1,:) = ...
                fminunc(@(x) myEnergy(x,...
                    [phat(j,:);o;D{i}(ind,4:5)],... % position
                    [vhat(j,:);zeros(size(o));D{i}(ind,6:7)],... % speed
                    u(j+1),...                          % ui
                    pdest(dhat(j+1),:),...              % zi
                    theta,...                           % params
                    [true;false(size(o,1),1);groups]),...   % group indicator
                vhat(j,:),...                           % init value
                optimset('GradObj','on',...
                         'LargeScale','off',...
                         'Display','off'...
                    ));
        elseif strcmp(method,'attrmc')
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
                    class),...                          % class
                vhat(j,:),...                           % init value
                optimset('GradObj','on',...
                         'LargeScale','off',...
                         'Display','off'...
                    ));
        elseif strcmp(method,'lin') % Linear
            vhat(j+1,:) = vhat(j,:);
        end
        phat(j+1,:) = phat(j,:) + dt*vhat(j+1,:);  % position
    end
    
    %% Compute euclid distance
    err{i} = sqrt(sum((p(2:end,:) - phat(2:end,:)).^2,2));
    % Save simulated info
    if Nargout > 1
        Pred{i} = D{i}(D{i}(:,2)>s(4)&D{i}(:,3)==pid,1:9);
        if ~isempty(Pred{i})
            Pred{i}(:,4:5) = phat(2:end,:);
            Pred{i}(:,6:7) = vhat(2:end,:);
            Pred{i}(:,8) = dhat(2:end);
        end
        Grp{i} = cat(1,g{:});
    end
end

% Reshape output
err = cat(1,err{:});
if Nargout > 1
    Pred = cat(1,Pred{:});
    Grp = cat(1,Grp{:});
end

end