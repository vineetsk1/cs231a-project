function [ D, T, Obj ] = importData()
%IMPORTDATA load and preprocess datasets for the pedestrian project

%% Load ewap dataset
ewap_paths = {...
    'ewap_dataset/seq_eth/',...
    'ewap_dataset/seq_hotel/',...
    };
ewap_labels = {'eth'...
    ,'hotel'...
    };
for did = 1:length(ewap_paths)
    D(did).label = ewap_labels{did};
    if exist('mmreader','class')
        D(did).video = [];%VideoReader([ewap_paths{did} 'video.avi']);
    end
    D(did).H = load([ewap_paths{did} 'H.txt']);
    % Obstacles
    P = load([ewap_paths{did} 'static.txt']);
    P = [P(:,[2 1]) ones(size(P,1),1)] * D(did).H';
    P = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
    D(did).obstacles = P;
    % Persons, Observations, Destinations
    [D(did).persons,D(did).observations,D(did).destinations] =...
        ewapObsmat2Tab([ewap_paths{did} 'obsmat.txt'],...
        [ewap_paths{did} 'destinations.txt'],...
        [ewap_paths{did} 'groups.txt']);
end

%% load ucy dataset
ucy_paths = {...
    'ucy_crowd/data_zara01/',...
    'ucy_crowd/data_zara02/',...
    'ucy_crowd/data_students03/',...
    };
ucy_labels = {'zara01','zara02','students03'};
if exist('mmreader','class')
    n = length(ewap_paths);
    for did = 1:length(ucy_paths)
        D(n+did).label = ucy_labels{did};
        D(n+did).video = VideoReader([ucy_paths{did} 'video.avi']);
        D(n+did).H = load([ucy_paths{did} 'H.txt']);
        % Obstacles
        P = load([ucy_paths{did} 'static.txt']);
        P = [P(:,[2 1]) ones(size(P,1),1)] * D(n+did).H';
        P = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
        D(n+did).obstacles = P;
        % Persons, Observations, Destinations
        [D(n+did).persons,D(n+did).observations,D(n+did).destinations] =...
            ucyObsmat2Tab([ucy_paths{did} 'annotation.vsp'],...
            [ucy_paths{did} 'destinations.txt'],...
            [ucy_paths{did} 'groups.txt'],...
            D(n+did).H,D(n+did).video);
    end
end


%load stanford
n=5;
obsmat=load('stanfobsmat.txt.txt');
stanford_label={'stanford_cross'};
obsmat=sortrows(obsmat,1);
dest=[720 170;400 404; 400 10; 10 170];
%n = length(ewap_paths)+length(ucy_paths);
did=1;
D(n+did).labelpers=stanford_label(did);
D(n+did).obstacles=[0 0];
D(n+did).video=[];
D(n+did).H = eye(3);
persons = unique(obsmat(:,2));
persons = [persons zeros(size(persons,1),3)];
observations = [obsmat(:,[1 2 3 5 6 8]) ones(size(obsmat,1),1)];
for i = 1:size(persons,1)
    % Select tuples of person i (time,id,p_x,p_y,v_x,v_y)
    ind = find(obsmat(:,2)==persons(i,1));
    t = obsmat(ind,[1 2 3 5 6 8]);
    % Basic measurement
    pprev = [t(1,[3 4]); t(1:end-1,[3 4])];
    % Destination selection
    % Cosine
    phi1 = atan2(t(end,4)-t(1,4),t(end,3)-t(1,3)); % angle of start to end
    phi2 = atan2(dest(:,2)-t(1,4),dest(:,1)-t(1,3)); % angle to goal
    d1 = [cos(phi2) sin(phi2)]*[cos(phi1);sin(phi1)];
    % Euclid
    d2 = (dest(:,1)-t(end,3)).^2 + (dest(:,2)-t(end,4)).^2;
    d2(d1<0) = inf;
    destid = find(d2==min(d2),1); % Choose closest dest in the same dir
    % Desired speed
    u = mean(sqrt(sum(t(:,[5 6]).^2,2)));
    % Group id
    gid = 0;
    % Update rows
    persons(i,2:4) = [destid u gid];
    observations(ind,5:6) = 2.5*(t(:,[3 4])-pprev); % velocity in m/s
    observations(ind([1 end]),7) = false;  % Init does not have correct velocity
end
D(n+did).persons=persons;
D(n+did).observations=observations;
D(n+did).destinations=dest;
% %% load Railway Lausanne dataset
%
% rail_paths = {...
%     'RailwayLausanne/',...
%     };
% rail_labels = {'hall'};
% if exist('mmreader','class')
%     n = length(ewap_paths)+length(ucy_paths);
%     for did = 1:length(rail_paths)
%         D(n+did).H = load([ewap_paths{did} 'H.txt']);
%         % Obstacles
%         P = load([ewap_paths{did} 'static.txt']);
%         P = [P(:,[2 1]) ones(size(P,1),1)] * D(n+did).H';
%         P = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
%         D(n+did).obstacles = P;
%         % Persons, Observations, Destinations
%         [D(n+did).persons,D(n+did).observations,D(n+did).destinations] =...
%             ewapObsmat2Tab([rail_paths{did} 'obsmat.txt'],...
%                            [rail_paths{did} 'destinations.txt'],...
%                            [rail_paths{did} 'groups.txt']);
%     end
% end
% mx 720 my 404 miy10 mix10
% y170   x400
% % Convert struct array to table format





[T,Obj] = d2tab(D);

end

% function [persons, observations, dest] = railObsmat2Tab( obsmat, dest)
%
% %RAILOBSMAT2TAB
% % Obsmat(t,id,px,pz,py,vx,vz,vy)
% % Dest(px,py)
%
% % Observations(t,id,px,py,vx,vy,valid)
% % Persons(pid,destid,u,gid)
% obsmat = load(obsmat);
% dest = load(dest);
%
% persons = unique(obsmat(:,2));
% persons = [persons zeros(size(persons,1),3)];
% observations = [obsmat(:,[1 2 3 5 6 8]) ones(size(obsmat,1),1)];
%
% % Load groups(gid,pid)
% groups = groupLoad(persons,groups);
%
% % List of pedestrians
% for i = 1:size(persons,1)
%     % Select tuples of person i (time,id,p_x,p_y,v_x,v_y)
%     ind = find(obsmat(:,2)==persons(i,1));
%     t = obsmat(ind,[1 2 3 5 6 8]);
%     % Basic measurement
%     pprev = [t(1,[3 4]); t(1:end-1,[3 4])];
%
%     % Destination selection
%     % Cosine
%     phi1 = atan2(t(end,4)-t(1,4),t(end,3)-t(1,3)); % angle of start to end
%     phi2 = atan2(dest(:,2)-t(1,4),dest(:,1)-t(1,3)); % angle to goal
%     d1 = [cos(phi2) sin(phi2)]*[cos(phi1);sin(phi1)];
%     % Euclid
%     d2 = (dest(:,1)-t(end,3)).^2 + (dest(:,2)-t(end,4)).^2;
%     d2(d1<0) = inf;
%     destid = find(d2==min(d2),1); % Choose closest dest in the same dir
%
%     % Desired speed
%     u = mean(sqrt(sum(t(:,[5 6]).^2,2)));
%
%     % Group id
%     gid = groups(groups(:,2)==persons(i),1);
%
%     % Update rows
%     persons(i,2:4) = [destid u gid];
%     observations(ind,5:6) = 2.5*(t(:,[3 4])-pprev); % velocity in m/s
%     observations(ind([1 end]),7) = false;  % Init does not have correct velocity
% end
%
% end


function [ persons, observations, dest ] = ewapObsmat2Tab( obsmat, dest, groups )
%EWAPOBSMAT2TAB
% Obsmat(t,id,px,pz,py,vx,vz,vy)
% Dest(px,py)

% Observations(t,id,px,py,vx,vy,valid)
% Persons(pid,destid,u,gid)
obsmat = load(obsmat);
dest = load(dest);

persons = unique(obsmat(:,2));
persons = [persons zeros(size(persons,1),3)];
observations = [obsmat(:,[1 2 3 5 6 8]) ones(size(obsmat,1),1)];

% Load groups(gid,pid)
groups = groupLoad(persons,groups);

% List of pedestrians
for i = 1:size(persons,1)
    % Select tuples of person i (time,id,p_x,p_y,v_x,v_y)
    ind = find(obsmat(:,2)==persons(i,1));
    t = obsmat(ind,[1 2 3 5 6 8]);
    % Basic measurement
    pprev = [t(1,[3 4]); t(1:end-1,[3 4])];
    
    % Destination selection
    % Cosine
    phi1 = atan2(t(end,4)-t(1,4),t(end,3)-t(1,3)); % angle of start to end
    phi2 = atan2(dest(:,2)-t(1,4),dest(:,1)-t(1,3)); % angle to goal
    d1 = [cos(phi2) sin(phi2)]*[cos(phi1);sin(phi1)];
    % Euclid
    d2 = (dest(:,1)-t(end,3)).^2 + (dest(:,2)-t(end,4)).^2;
    d2(d1<0) = inf;
    destid = find(d2==min(d2),1); % Choose closest dest in the same dir
    
    % Desired speed
    u = mean(sqrt(sum(t(:,[5 6]).^2,2)));
    
    % Group id
    gid = groups(groups(:,2)==persons(i),1);
    
    % Update rows
    persons(i,2:4) = [destid u gid];
    observations(ind,5:6) = 2.5*(t(:,[3 4])-pprev); % velocity in m/s
    observations(ind([1 end]),7) = false;  % Init does not have correct velocity
end

end



function [ persons, observations, dest ] = ucyObsmat2Tab( annotation, dest, groups, H, video )
%EWAPOBSMAT2TAB
% Obsmat(t,id,px,pz,py,vx,vz,vy)
% Dest(px,py)

% Observations(t,id,px,py,vx,vy,valid)
% Persons(pid,destid,u,gid)

dest = load(dest);

% Read annotation file: Tr{pid}(px,py,time,head)
fid = fopen(annotation,'r');
nTr = fscanf(fid,'%d - the number of splines\n',1); % header
Tr = cell(nTr,1);
for i = 1:nTr
    nCP = fscanf(fid,'%d - Num of control points\n',1);
    Tr{i} = fscanf(fid,'%f %f %d %f - (2D point, m_id)\n',[4 nCP])';
end
fclose(fid);

% Decide duration
t_s = min(cellfun(@(x) min(x(:,3)),Tr));
t_e = max(cellfun(@(x) max(x(:,3)),Tr));
frames = (t_s:(video.FrameRate/2.5):t_e)';

% Interpolation
D = cell(size(Tr));
for i = 1:length(D)
    duration = frames(min(Tr{i}(:,3))<=frames&...
        max(Tr{i}(:,3))>=frames);
    D{i} = [(duration+1)...                   % timestamp
        i*ones(size(duration))...         % person id
        interp1(Tr{i}(:,3),Tr{i}(:,2),duration)... % py
        interp1(Tr{i}(:,3),Tr{i}(:,1),duration)... % px
        ones(size(duration,1),3)...       % memory alloc for v, flag
        ];
end
D = cat(1,D{:});

% Shift image coordinates
D(:,3) =  -D(:,3) + video.Height/2;
D(:,4) =  D(:,4) + video.Width/2;
if ~isempty(strfind(annotation,'zara02')) % bugfix for zara02
    D(:,3) = D(:,3) + 50; % Move annotation from head to foot
end
% Homography transform
P = [D(:,3:4) ones(size(D,1),1)] * H';
D(:,3:4) = [P(:,1)./P(:,3) P(:,2)./P(:,3)];

% Make tables
groups = groupLoad((1:nTr)',groups);
persons = [(1:nTr)' zeros(nTr,3)];
observations = D;

% Populate columns
for i = 1:size(persons,1)
    % Select tuples of person i (time,id,p_x,p_y,v_x,v_y)
    ind = find(observations(:,2)==persons(i,1));
    t = observations(ind,:);
    % Basic measurement
    v = 2.5*(t(2:end,[3 4]) - t(1:end-1,[3 4]));
    
    % Destination selection
    % Cosine
    phi1 = atan2(t(end,4)-t(1,4),t(end,3)-t(1,3)); % angle of start to end
    phi2 = atan2(dest(:,2)-t(1,4),dest(:,1)-t(1,3)); % angle to goal
    d1 = [cos(phi2) sin(phi2)]*[cos(phi1);sin(phi1)];
    % Euclid
    d2 = (dest(:,1)-t(end,3)).^2 + (dest(:,2)-t(end,4)).^2;
    d2(d1<0) = inf;
    destid = find(d2==min(d2),1); % Choose closest dest in the same dir
    
    % Desired speed
    u = mean(sqrt(sum(v.^2,2)));
    
    % Group id
    gid = groups(groups(:,2)==persons(i),1);
    
    % Update rows
    persons(i,2:4) = [destid u gid];
    observations(ind,5:6) = [v(1,:);v]; % velocity in m/s
    observations(ind(1),7) = false;  % Init does not have correct velocity
end

end

function [ G ] = groupLoad( V, inputfile )
%GROUPLOAD return group-person table
% G(gid,pid): table of gid-pid relationship
% V(pid): list of pid
% inputfile: text file containing in each line: pid pid pid...

% Load non-empty groups
E = zeros(0,2);
fid = fopen(inputfile,'r');
while ~feof(fid)
    l = sscanf(fgetl(fid),'%d');
    if ~isempty(l)
        E = [E;combnk(l,2)];
    end
end
fclose(fid);

% Find connected components
mark = true(size(V)); % indicator for unvisited nodes
C = {};
k = 1;
while any(mark)
    first = find(mark,1);
    mark(first) = false;
    C{k} = unique([V(first)...
        E(E(:,1)==V(first),2)'...
        E(E(:,2)==V(first),1)']);
    ind = arrayfun(@(x)any(x==C{k}),V);
    while any(mark(ind))
        mark(ind) = false;
        C{k} = unique([C{k}...
            E(arrayfun(@(x)any(x==C{k}),E(:,1)),2)'...
            E(arrayfun(@(x)any(x==C{k}),E(:,2)),1)']);
        ind = arrayfun(@(x)any(x==C{k}),V);
    end
    k = k + 1;
end

% Create table structure
G = cell2mat(arrayfun(@(x) [x*ones(length(C{x}),1) C{x}'],...
    (1:length(C))','UniformOutput',false));

end

function [ T, Obj ] = d2tab( D )
%D2TAB convert data struct to table format
% T(dataset,time,pid,px,py,vx,vy,pxnext,pynext,pxdext,pydest,u,flag,gid)
% Obj(dataset,px,py)

T = cell(length(D),1);
Obj = cell(length(D),1);
for i = 1:length(D)
    % Query related data columns
    pnext = zeros(size(D(i).observations,1),2);
    pdest = zeros(size(D(i).observations,1),2);
    u = zeros(size(D(i).observations,1),1);
    gid = zeros(size(D(i).observations,1),1);
    last = false(size(D(i).observations,1),1);
    for j = 1:size(D(i).persons,1)
        % Query pnext
        ind = find(D(i).observations(:,2)==D(i).persons(j,1));
        if isempty(ind), keyboard; end
        t = D(i).observations(ind,3:4);
        pnext(ind,:) = [t(2:end,:);t(end,:)];
        % Query pdest
        pdest(ind,1) = D(i).destinations(D(i).persons(j,2),1);
        pdest(ind,2) = D(i).destinations(D(i).persons(j,2),2);
        % Query u
        u(ind) = D(i).persons(j,3);
        % Query gid
        gid(ind) = D(i).persons(j,4);
        % Mark last
        last(ind(end)) = true;
    end
    % Update flag
    flag = D(i).observations(:,7);
    flag(last) = false;
    % Store
    T{i} = [i*ones(size(D(i).observations,1),1) ... % did
        D(i).observations(:,1:6) ...            % t,pid,px,py,vx,vy
        pnext pdest u flag gid];
    Obj{i} = [i*ones(size(D(i).obstacles,1),1) ... % did
        D(i).obstacles];  % px py
end
T = cat(1,T{:});
Obj = cat(1,Obj{:});

end