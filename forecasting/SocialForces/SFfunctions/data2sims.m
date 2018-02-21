function [ S ] = data2sims( D, varargin )
%DATA2SIMS generates simulation struct from dataset
%
% Input:
%   D: dataset struct each containing
%          label: label of the dataset
%          video: video object
%              H: homography
%      obstacles: obstacles (px,py)
%        persons: persons (id,dest,u,gid)
%   observations: observations (t,id,px,py,vx,vy,flag)
%   destinations: destinations (px,py)
% Output:
%   S: array of simulation struct each containing
%        dataset: label of the dataset
%            vid: video object
%         frames: timestamp
%         offset: number of offset frames/observations before start
%              H: homography
%           obst: obstacles (px,py)
%           dest: destinations (px,py)
%           trks: tracker instances (id,start,end)
%           obsv: observations (t,id,px,py,vx,vy,dest,u)
%           grps: group relations (t,id,id,label)

%% Config
sampleMethod = 'mult';  % Method of sampling, either 'single' or 'mult' 
Noffset = 5;            % Frame offset before starting the simulation
Ninterval = inf;        % Interval of the simulation
Nduration = inf;        % Duration of the simulation
Datasets = 1:length(D); % All datasets are converted
Ooffset = 10;           % Offset before the first tracker for bg and svm
Toffset = 1;            % Offset before each tracker for valid velocity
Cd = cell(length(D),1); % Destination classifiers
Cg = [];                % Group classifier
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'SampleMethod'), sampleMethod = varargin{i+1}; end
    if strcmp(varargin{i},'Noffset'), Noffset = varargin{i+1}; end
    if strcmp(varargin{i},'Ninterval'), Ninterval = varargin{i+1}; end
    if strcmp(varargin{i},'Nduration'), Nduration = varargin{i+1}; end
    if strcmp(varargin{i},'Datasets'), Datasets = varargin{i+1}; end
    if strcmp(varargin{i},'Ooffset'), Ooffset = varargin{i+1}; end
    if strcmp(varargin{i},'Toffset'), Toffset = varargin{i+1}; end
    if strcmp(varargin{i},'DestClassifier'), Cd = varargin{i+1}; end
    if strcmp(varargin{i},'GroupClassifier'), Cg = varargin{i+1}; end
end

%% Create subsampling index
IDX = cell(length(Datasets),1);
for i = Datasets
    if strcmp(sampleMethod,'single')
        % Decides seeds by (time,person) pair
        persons = D(i).persons(:,1);
        I = cell(length(persons),1);
        for j = 1:length(persons)
            frames = D(i).observations(...
                D(i).observations(:,2)==persons(j),1);
            ninterval = Ninterval;
            ninterval(isinf(ninterval)) = length(frames);
            starts = (1+Noffset:ninterval:length(frames));
            if isempty(starts)
                I{j} = zeros(0,5);
            else
                % Decides end of simulation
                ends = min(length(frames),starts+Nduration);
                I{j} = [i*ones(length(starts),1)...
                        frames(starts) frames(ends)...
                        frames(max(1,starts-Ooffset))...
                        persons(j)*ones(length(starts),1)];
            end
        end
        % Save (simulation,start,end,start-Ooffset,person)
        IDX{i} = cat(1,I{:});
    else
        % Decides seeds by time
        frames = unique(D(i).observations(:,1));
        ninterval = Ninterval;
        ninterval(isinf(ninterval)) = length(frames);
        starts = (1+Noffset:ninterval:length(frames));
        % Decides end of simulation
        ends = min(length(frames),starts+Nduration);
        % Save (simulation,start,end,start-Ooffset,person)
        IDX{i} = [i*ones(length(starts),1)...
                  frames(starts) frames(ends)...
                  frames(max(1,starts-Ooffset))...
                  zeros(length(starts),1)];
    end
end
IDX = cat(1,IDX{:});

%% Subsample sequence for simulation
for i = 1:size(IDX,1)
    % Trackers
    persons = unique(D(IDX(i,1)).observations(:,2))';
    if IDX(i,5)==0  % If person id is set zero, include everyone
        subjects = persons;
    else
        subjects = IDX(i,5);
    end
    trackers = cell(length(subjects),1);
    for j = 1:length(trackers)
        % Query frames where that person is appearing
        frames = D(IDX(i,1)).observations(...
            D(IDX(i,1)).observations(:,1)>=IDX(i,2)&... % after start
            D(IDX(i,1)).observations(:,1)<=IDX(i,3)&... % before end
            D(IDX(i,1)).observations(:,2)==subjects(j),1);
        % TODO: discontinuity detection
        % TODO: cropping
        % Keep persons with valid number of observations
        if length(frames) <= Toffset
            trackers{j} = zeros(0,3);
        else
            trackers{j} = [subjects(j) frames(Toffset) frames(end)];
        end
    end
    trackers = cat(1,trackers{:});
    
    % Observations
    obsv = cell(length(persons),1);
    for j = 1:length(persons)
        % Query observations where that person is appearing
        X = D(IDX(i,1)).observations(...
            D(IDX(i,1)).observations(:,1)>=IDX(i,4)&...% after start-offset
            D(IDX(i,1)).observations(:,1)<=IDX(i,3)&...% before end
            D(IDX(i,1)).observations(:,2)==persons(j),1:6); % t,id,p,v
        % Query persons comfortable speed, destination label, and gid
        Y = D(IDX(i,1)).persons(D(IDX(i,1)).persons(:,1)==persons(j),2:4);
        % Save joined structure (t,id,px,py,vx,vy,dest,u,gid)
        obsv{j} = [X repmat(Y,[size(X,1) 1])];
    end
    obsv = cat(1,obsv{:});
    
    % Groups
    frames = unique(D(IDX(i,1)).observations(...
        D(IDX(i,1)).observations(:,1)>=IDX(i,2)&...
        D(IDX(i,1)).observations(:,1)<=IDX(i,3),1))';
    grps = cell(length(frames),1);
    for j = 1:length(frames)
        % Query persons at frame j
        persons = obsv(obsv(:,1)==frames(j),2);
        if length(persons) < 2
            grps{j} = zeros(0,4);
        else
            % Construct tables and labels
            [I1 I2] = meshgrid(persons);
            ind = I1~=I2; I1 = I1(ind); I2 = I2(ind);   % Drop self loops
            Y = arrayfun(@(k)...
                obsv(obsv(:,1)==frames(j)&obsv(:,2)==I1(k),9)==...
                obsv(obsv(:,1)==frames(j)&obsv(:,2)==I2(k),9),...
                1:length(I1))';
            % grps(t,pid,pid,label)
            grps{j} = [repmat(frames(j),[length(I1) 1]) I1 I2 Y];
        end
    end
    grps = cat(1,grps{:});
    obsv(:,end) = []; % drop gid as it's not static
    
    % Create struct array
    S(i).dataset = D(IDX(i,1)).label;
    S(i).vid = D(IDX(i,1)).video;
    S(i).frames = unique(D(IDX(i,1)).observations(...
        D(IDX(i,1)).observations(:,1)>=IDX(i,4) &...
        D(IDX(i,1)).observations(:,1)<=IDX(i,3),1))';
    S(i).offset = Ooffset;
    S(i).H = D(IDX(i,1)).H;
    S(i).obst = D(IDX(i,1)).obstacles;
    S(i).dest = D(IDX(i,1)).destinations;
    S(i).trks = trackers;
    S(i).obsv = obsv;
    S(i).grps = grps;
    S(i).Cd = Cd{IDX(i,1)};
    S(i).Cg = Cg;
end

S(arrayfun(@(s) isempty(s.trks),S)) = [];

end

