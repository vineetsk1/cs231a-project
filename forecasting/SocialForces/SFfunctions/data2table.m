function [ Obsv, Obst, Dest ] = data2table( D, varargin )
%DATA2TABLE converts dataset to table format
%
% Input:
%   D: data struct
% Output:
%   Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Obst(dataset,px,py)
%   Dest(dataset,px,py)

Obsv = cell(length(D),1);
Obst = cell(length(D),1);
Dest = cell(length(D),1);
for i = 1:length(D)
    % Join persons table to observations
    dest = zeros(size(D(i).observations,1),1);
    u = zeros(size(D(i).observations,1),1);
    gid = zeros(size(D(i).observations,1),1);
    for j = 1:size(D(i).persons,1)
        ind = D(i).observations(:,2)==D(i).persons(j,1);
        % Query destination id
        dest(ind,1) = D(i).persons(j,2);
        % Query u
        u(ind) = D(i).persons(j,3);
        % Query gid
        gid(ind) = D(i).persons(j,4);
    end
    % Don't forget flag
    flag = D(i).observations(:,7);
    % Store
    Obsv{i} = [i*ones(size(D(i).observations,1),1) ... % did
               D(i).observations(:,1:6) ...            % t,pid,px,py,vx,vy
               dest u gid flag];                       % dest,u,group,flag
    Obst{i} = [i*ones(size(D(i).obstacles,1),1) ... % did
               D(i).obstacles];  % px py
    Dest{i} = [i*ones(size(D(i).destinations,1),1) ... % did
               D(i).destinations];  % px py
end
Obsv = cat(1,Obsv{:});
Obst = cat(1,Obst{:});
Dest = cat(1,Dest{:});

end

