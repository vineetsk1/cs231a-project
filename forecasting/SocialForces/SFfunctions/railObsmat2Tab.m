function [persons, observations, dest] = railObsmat2Tab( obsmat, dest)

%RAILOBSMAT2TAB
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