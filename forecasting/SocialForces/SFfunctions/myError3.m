function [ err ] = myError3( T, Obj, params, varargin )
%MYERROR3 computes error in n-step prediction given data
%
%   [err] = myError3(T,params)
%
% Input:
%   T: Table(dataset,t,id,px,py,vx,vy,pnextx,pnexty,pdestx,pdesty,u,valid,gid)
%   params: 1-by-6 parameter vector
% Output:
%   err : N-by-1 vector of errors in position prediction

Ninterval = 3;  % Interval to start simulation
Nstep = 12;     % Steps to predict
Npast = 100;      % Past information available at simulation
Tind = unique(T(:,[1 3]),'rows');
MODEL = [];
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Interval'),  Ninterval = varargin{i+1};  end
    if strcmp(varargin{i},'Step'),      Nstep = varargin{i+1};      end
    if strcmp(varargin{i},'Index'),     Tind = varargin{i+1};       end
    if strcmp(varargin{i},'GroupEstimate')
        MODEL = varargin{i+1};
    end
end

%% Arrange data structure
% We will store necessary data for simulation in cell array D
% Each array element will be then used in parfor loop for evaluation

% Decide the seeds for simulation
seeds = false(size(T,1),1);
for i = 1:size(Tind,1) % for each (datasets,person)
    ind = find(T(:,1)==Tind(i,1)&T(:,3)==Tind(i,2)&T(:,13)==1);
    seeds(ind(1:Ninterval:length(ind))) = true; % mark the seed
end
seeds = find(seeds'); % Make it sparse

% Create a cell array for each seed
D = cell(length(seeds),1);
start = zeros(length(seeds),1);
err = cell(length(seeds),1);
for i = 1:length(seeds)
    % Query index info
    did = T(seeds(i),1); % dataset id
    t_s = T(seeds(i),2); % first timestamp
    pid = T(seeds(i),3); % person id
    % Decide duration of simulation
    time = T(T(:,1)==did&T(:,2)>=t_s&T(:,3)==pid,2);
    time = time(1:min(length(time),Nstep));  % duration of simulation
    t_e = time(end);
    % Save everything necessary for simulation in the array
    D{i} = [T(T(:,1)==did&T(:,2)<=t_e&T(:,3)==pid,:);...
            T(T(:,1)==did&T(:,2)<=t_e&T(:,3)~=pid,:)];
    start(i) = t_s;
    err{i} = zeros(length(time)-1,1);
end

%% Compute prediction error
parfor i = 1:length(err)
    % Query time duration for simulation
    time = unique(D{i}(D{i}(:,2)>=start(i),2))';
    % Query obstacle location
    o = Obj; o = o(o(:,1)==D{i}(1,1),2:3);
    % Simulate the behavior of person pid
    pid = D{i}(1,3);
    p = D{i}(D{i}(:,2)>=start(i)&D{i}(:,3)==pid,4:5); % list of positions
    v = D{i}(D{i}(:,2)==start(i)&D{i}(:,3)==pid,6:7); % initial velocity
    ui = D{i}(1,12);                % preferred speed
    zi = D{i}(1,10:11);             % destination
    phat = zeros(size(p));          % list of predicted positions of pid
    phat(1,:) = p(1,:);             % initial position = grand truth
    for j = 1:length(time)-1
        % Estimate groups
        ind = find(D{i}(:,2)==time(j)&D{i}(:,3)~=pid);
        past = D{i}(D{i}(:,2)<=time(j),2)';
        past = past(max(1,length(past)-Npast+1));
        if ~isempty(MODEL)
            X = zeros(length(ind),grpFeature('len'));
            for k = 1:length(ind)
                X(k,:) = grpFeature(D{i}(D{i}(:,2)>=past &...
                                         D{i}(:,2)<=time(j)&...
                                         D{i}(:,3)==pid,[2 4:7]),...
                                    D{i}(D{i}(:,2)>=past &...
                                         D{i}(:,2)<=time(j)&...
                                         D{i}(:,3)==D{i}(ind(k),3),[2 4:7]));
            end
            g = logical(svmpredict(...
                double(D{i}(ind,14)==D{i}(1,14)),X,MODEL));
        else
            g = D{i}(ind,14)==D{i}(1,14);
        end
        % Compute optimal velocity
        v = fminunc(@(x) myEnergy(x,...
                    [phat(j,:);o;D{i}(ind,4:5)],...     % position
                    [v;zeros(size(o));D{i}(ind,6:7)],...% speed
                    ui,...                              % ui
                    zi,...                              % zi
                    params,...                          % params
                    [true;false(size(o,1),1);g]...      % group
                    ),...
                v,...                        % init value
                optimset('GradObj','on',...
                    'LargeScale','off',...
                    'Display','off'...
                    ));
        % Predict the next position
        phat(j+1,:) = phat(j,:) + 0.4*v;
    end
    % Compute euclid distance
    err{i} = sqrt(sum((p(2:end,:) - phat(2:end,:)).^2,2));
end

% Reshape the output
err = cat(1,err{:});

end
