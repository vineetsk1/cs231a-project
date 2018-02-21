function [ groups, M, A ] = myEstimate( T, Others, params )
%EWAPERROR computes error measure in group correspondence
%
%   [A] = ewapError2(T,params)
%
% Input:
%   T: Table(dataset,t,id,px,py,vx,vy,pnextx,pnexty,pdestx,pdesty,u)
%   Others: cell array of associated tuples (id,px,py,vx,vy,groups) of other
%           pedestrians
%   params: 1-by-6 parameter vector
% Output:
%   err : N-by-1 vector of errors for each row

threshold = 0.01;
mindist = 3.0;

%% Estimate attracting pedestrians for each observation
A = cell(size(T,1),1);
parfor i = 1:size(T,1)
    t = T(i,:); % Query tuple of data i
    theta = params;
    vhat = ((t([8 9]) - t([4 5]))*2.5 - theta(6)*t([6 7])) / (1-theta(6));
    % Create a list of possible attraction subset
    ind = find(t(6)*Others{i}(:,4)+t(7)*Others{i}(:,5)>0 &...
               sqrt((t(4)-Others{i}(:,2)).^2+(t(5)-Others{i}(:,3)).^2)<mindist)';
    g = cellfun(@(z) logical(str2double(z)),...
                     num2cell(dec2bin(0:2^length(ind)-1)));
    % Find the most likely attraction subset
    E = zeros(size(g,1),1);
    for j = 1:size(g,1)
        pat = false(size(Others{i},1),1);
        pat(ind(g(j,:))) = true;
        E(j) = ewapEnergy(vhat,...
                          [t([4 5]); Others{i}(:,[2 3])],...
                          [t([6 7]); Others{i}(:,[4 5])],...
                          t(12),...
                          t([10 11]),...
                          theta,...
                          logical([false; pat]));
    end
    g = g(E==min(E),:);
    if size(g,1) > 1, g = g(find(sum(g,2) == min(sum(g,2)),1),:); end
    
    % Compute the difference between truth and prediction in position
    A{i} = Others{i}(g,1);
end

%% Aggregate attraction over trajectories
datasets = unique(T(:,1))';
M = cell(1,length(datasets));
M0 = cell(1,length(datasets));
groups = cell(1,length(datasets));
for d_id = datasets
    % Select records from dataset d_id
    t = T(T(:,1)==d_id,:);
    persons = unique(t(:,3))';
    M{d_id} = zeros(length(persons));
    for p_id = persons
        % Compute normalized frequency of person j in the set s
        s = (cat(1,A{T(:,1)==d_id & T(:,3)==p_id}));
        others_id = unique(s);
        h = histc(s,others_id) / nnz(t(:,3)==p_id);
        % Save
        for j = 1:length(others_id)
            M{d_id}(persons==p_id,persons==others_id(j)) = h(j);
        end
    end
    % Filter bidirectional interaction and return connected components
    m = (M{d_id}.*M{d_id}') > threshold;
    groups{d_id} = cellfun(@(x) persons(x),...
        conn(m|eye(length(persons))),...
        'UniformOutput',false);
    
    % Get grand truth
    M0{d_id} = false(length(persons),length(persons));
    for p_id = persons
        % Compute normalized frequency of person j in the set s
        s = (cat(1,Others{T(:,1)==d_id & T(:,3)==p_id}));
        others_id = unique(s(s(:,6)>0,1)');
        % Save
        for j = 1:length(others_id)
            M0{d_id}(persons==p_id,persons==others_id(j)) = true;
        end
    end
    M0{d_id} = M0{d_id} | M0{d_id}';
    fprintf('Dataset %d\n',d_id);
    fprintf('  Recall:    %f\n',nnz(m & M0{d_id}) / nnz(M0{d_id}));
    fprintf('  Precision: %f\n',nnz(m & M0{d_id}) / nnz(m));
end


end

function [ C ] = conn( M )
%CONN return connected components of adjecency matrix

ind = true(1,size(M,1));
ind(~any(M,2)) = false;
C = {}; k = 1;
while any(ind)
    first = find(ind,1);
    ind(first) = false;
    C{k} = find(M(first,:));
    while any(ind(C{k}))
        ind(C{k}) = false;
        C{k} = unique([C{k} find(any(M(C{k},:),1))]);
    end
    k = k + 1;
end
end
