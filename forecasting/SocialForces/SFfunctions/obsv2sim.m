function [ Obs, Sims ] = obsv2sim( Obsv, varargin )
%OBSV2SIM subsamples simulation sequence and expands observation table
%
% Input:
%   Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
% Output:
%   Obs(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Sims(simulation,dataset,person,start,duration)

Ninterval = 3;  % Interval to subsample simulation sequence
Npast = 1;      % Past steps to include
Nduration = 12; % Future steps to include (= #simulation steps)
Tind = unique(Obsv(:,[1 3]),'rows');    % Index of (dataset,person)
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Interval'),  Ninterval = varargin{i+1};  end
    if strcmp(varargin{i},'Duration'),  Nduration = varargin{i+1};  end
    if strcmp(varargin{i},'Past'),      Npast = varargin{i+1};      end
    if strcmp(varargin{i},'Index'),     Tind = varargin{i+1};       end
end

% Decide the seeds for simulation
seeds = false(size(Obsv,1),1);
for i = 1:size(Tind,1) % for each (datasets,person)
    ind = find(Obsv(:,1)==Tind(i,1)&Obsv(:,3)==Tind(i,2)&Obsv(:,11)==1);
    seeds(ind(1:Ninterval:length(ind))) = true; % mark the seed
end
seeds = find(seeds); % Make it sparse

% Create a cell array for each seed
Obs = cell(length(seeds),1);
Sims = zeros(length(seeds),5);
for i = 1:length(seeds)
    % Query index info
    did = Obsv(seeds(i),1); % dataset id
    t_s = Obsv(seeds(i),2); % timestamp of seed
    pid = Obsv(seeds(i),3); % person id
    % Set time range to include
    time = Obsv(Obsv(:,1)==did&Obsv(:,3)==pid,2);
    c = find(time==t_s);
    time = time(max(1,c-Npast):c+min(nnz(time>t_s),Nduration));
    % Save everything necessary for simulation in the array
    ind = find(Obsv(:,1)==did&Obsv(:,2)>=time(1)&Obsv(:,2)<=time(end));
    Obs{i} = [i*ones(length(ind),1) Obsv(ind,2:end)];
    Sims(i,:) = [i did pid t_s min(nnz(time>t_s),Nduration)];
end
Obs = cat(1,Obs{:});

end

