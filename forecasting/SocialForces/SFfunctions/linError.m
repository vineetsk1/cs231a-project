function [ err ] = linError( T, varargin )
%LINERROR error evaluation for linear predictor
%   Detailed explanation goes here

% Config
Ninterval = 3;
Nstep = 12;
Tind = unique(T(:,[1 3]),'rows');
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Interval'),  Ninterval = varargin{i+1};  end
    if strcmp(varargin{i},'Step'),      Nstep = varargin{i+1};      end
    if strcmp(varargin{i},'Index'), Tind = varargin{i+1};   end
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
err = cell(length(seeds),1);
parfor i = 1:length(seeds)
    % Query index info
    did = T(seeds(i),1); % dataset id
    t_s = T(seeds(i),2); % first timestamp
    pid = T(seeds(i),3); % person id
    % Decide duration of simulation
    time = T(T(:,1)==did&T(:,2)>=t_s&T(:,3)==pid,2);
    time = time(1:min(length(time),Nstep));  % duration of simulation
    t_e = time(end);
    % Save everything necessary for simulation in the array
    D{i} = T(T(:,1)==did&T(:,2)>=t_s&T(:,2)<=t_e&T(:,3)==pid,:);
    err{i} = zeros(length(time)-1,1);
end

%% Compute error
for i = 1:length(err)
    % Query time duration for simulation
    time = unique(D{i}(:,2))';
    % Simulate the behavior of person pid
    pid = D{i}(1,3);
    p = D{i}(D{i}(:,3)==pid,4:5);   % list of positions of pid
    v = D{i}(1,6:7);                % initial velocity
    phat = zeros(size(p));          % list of predicted positions of pid
    phat(1,:) = p(1,:);             % initial position = grand truth
    for j = 1:length(time)-1
        % Predict position assuming v is static
        phat(j+1,:) = phat(j,:) + 0.4*v;
    end
    % Compute euclid distance
    err{i} = sqrt(sum((p(2:end,:) - phat(2:end,:)).^2,2));
end

% Reshape the output
err = cat(1,err{:});

end

