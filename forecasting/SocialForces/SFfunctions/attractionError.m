function [ err ] = attractionError( T, Obj, params, varargin )
%EWAPERROR2 computes norm of the partial derivative of individual energy
%
%   [ err ] = attractionError(T,theta)
%
% Input:
%   T: Table(dataset,t,id,px,py,vx,vy,pnextx,pnexty,pdestx,pdesty,u,valid,gid)
%   params: 1-by-6 parameter vector
% Output:
%   err : N-by-1 vector of errors for each row

Ninterval = 1;
maxdist = 3.5;  % Maximum distance to consider attraction effect
Tind = unique(T(:,[1 3]),'rows');
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'MaxDist'), maxdist = varargin{i+1}; end
    if strcmp(varargin{i},'Index'), Tind = varargin{i+1};   end
end

%% Arrange data structure
% We will store necessary data for one-step prediction in cell array D
% Each array element will be then used in parfor loop for evaluation

% Decide the seeds for evaluation
seeds = false(size(T,1),1);
for i = 1:size(Tind,1) % for each (datasets,person)
    ind = find(T(:,1)==Tind(i,1)&T(:,3)==Tind(i,2)&T(:,13)==1);
    ind = ind(arrayfun(@(j) nnz(T(:,2)==T(j,2))>1,ind));
    seeds(ind(1:Ninterval:length(ind))) = true; % mark the seed
end
seeds = find(seeds'); % Make it sparse

% Create a cell array for each seed
D = cell(length(seeds),1);
err = zeros(length(seeds),1);
for i = 1:length(seeds)
    % Query index info
    did = T(seeds(i),1); % dataset id
    t_s = T(seeds(i),2); % first timestamp
    pid = T(seeds(i),3); % person id
    % Save everything necessary for simulation in the array
    t = T(T(:,1)==did&T(:,2)==t_s&T(:,3)==pid,:);
    Others = T(T(:,1)==did&T(:,2)==t_s&T(:,3)~=pid,:);
    % index for other pedestrians to consider
    ind = find(t(6)*Others(:,6)+t(7)*Others(:,7)>0 &...
               sqrt((t(4)-Others(:,4)).^2+(t(5)-Others(:,5)).^2)<maxdist)';
    D{i} = [t;Others(ind,:)];
end

%% Compute prediction error
parfor i = 1:length(err) % parfor
    % Query obstacle location
    o = Obj; o = o(o(:,1)==D{i}(1,1),2:3);
    % Get vhat
    vstar = 2.5*(D{i}(1,8:9)-D{i}(1,4:5));
    % Create a list of possible attraction subset
    ind = (1:size(D{i},1)-1)';
    g = cellfun(@(z) logical(str2double(z)),...
                     num2cell(dec2bin(0:2^length(ind)-1)));
    % Find the most likely attraction subset
    E = zeros(size(g,1),1);
    for j = 1:size(g,1)
        pat = false(length(ind),1);
        pat(ind(g(j,:))) = true;
        % Compute optimal velocity choice
        E(j) = myEnergy(vstar,...
                        [D{i}(1,4:5); o; D{i}(2:end,4:5)],...% p
                        [D{i}(1,6:7); zeros(size(o));...
                         D{i}(2:end,6:7)],...           % v
                        D{i}(1,12),...                  % ui
                        D{i}(1,[10 11]),...             % zi
                        params,...
                        logical([true;false(size(o,1),1);pat])); % groups
    end
    g = g(E==min(E),:);
    if size(g,1) > 1, g = g(find(sum(g,2)==min(sum(g,2)),1),:); end

    % Compute the difference between truth and prediction in position
    truth = (D{i}(2:end,14)==D{i}(1,14))';
    err(i) = nnz(xor(g,truth));
end

end
