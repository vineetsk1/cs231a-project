function [ D, Obj ] = seq2ewap( seq )
%SEQ2EWAP compute necessary data and create data struct used in ewap
%
% [D,Obj] = seq2ewap(seq)
%
% Input:
%  seq: seqence struct of ewap dataset
% Output:
%  D: Table
%     (dataset,t,id,px,py,vx,vy,pnextx,pnexty,pdestx,pdesty,u,valid,gid)
%  Obj: Table (dataset,px,py)

% Merge datasets
datasets = fieldnames(seq);
D = cell(1,length(datasets));
Obj = cell(1,length(datasets));
for id = 1:length(datasets)
    [D{id},Obj{id}] = iseq2ewap(seq.(datasets{id}),id);
end
D = cat(1,D{:});
Obj = cat(1,Obj{:});

end

function [ T, Obj ] = iseq2ewap( seq, id )
%ISEQ2EWAP compute necessary data and create data struct used in ewap

% Dataset id (optional)
if ~exist('id','var'), id = 1; end

% Table (dataset,t,id,px,py,vx,vy,pnextx,pnexty,pdestx,pdesty,u,valid,gid)
T = [id * ones(size(seq.obsmat,1),1)...
     seq.obsmat(:,[1 2 3 5 6 8])...
     ones(size(seq.obsmat,1),6)...
     arrayfun(@(x) seq.groups(seq.groups(:,1)==x,2),seq.obsmat(:,2))];

% List of pedestrians
people = unique(seq.obsmat(:,2));
dest = seq.destinations;
for i = 1:length(people)
    % Select tuples of person i (time,id,p_x,p_y,v_x,v_y)
    ind = find(seq.obsmat(:,2) == people(i));
    t = seq.obsmat(ind,[1 2 3 5 6 8]);
    % Basic measurement
    pprev = [t(1,[3 4]); t(1:end-1,[3 4])];
    pnext = [t(2:end,[3 4]); t(end,[3 4])];
    % Destination discretization
    % Cosine
    phi1 = atan2(t(end,4)-t(1,4),t(end,3)-t(1,3)); % angle of start to end
    phi2 = atan2(dest(:,2)-T(1,4),dest(:,1)-t(1,3)); % angle to goal
    d1 = [cos(phi2) sin(phi2)]*[cos(phi1);sin(phi1)];
    % Euclid
    d2 = (dest(:,1)-t(end,3)).^2 + (dest(:,2)-t(end,4)).^2;
    d2(d1<0) = inf;
    pdest = repmat(dest(find(d2==min(d2),1),:),[length(ind) 1]);
%     pdest = repmat(t(end,[3 4]),[length(ind) 1]); % truth
    u = repmat(mean(sqrt(sum(t(:,[5 6]).^2,2))),[length(ind) 1]);
    % Update rows
    T(ind,8:12) = [pnext pdest u];       % populate columns
    T(ind,6:7) = 2.5*(t(:,[3 4])-pprev); % velocity in m/s
    T(ind(1),13) = false;                % Initial does not have velocity
    T(ind(end),13) = false;              % End does not have pnext
end

% Set up static objects
% [r,c] = find(seq.map);
% Obj = [r c ones(length(r),1)] * seq.H';
% Obj = Obj(1:100:size(Obj,1),:); % Make it sparse
Obj = [seq.static(:,[2 1]) ones(size(seq.static,1),1)] * seq.H';
Obj = [id*ones(size(Obj,1),1) Obj(:,1)./Obj(:,3) Obj(:,2)./Obj(:,3)];

end

