function [] = testGrp(varargin)
%TESTDEST evaluates quality of the destination prediction

% Config
Ninterval = 1;  % Subsample seed interval
Nduration = 1;  % Subsample duration
Nfold = 3;      % Cross validation
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Interval'), Ninterval = varargin{i+1}; end
    if strcmp(varargin{i},'Duration'), Nduration = varargin{i+1}; end
end
fprintf('Config\n');
fprintf('  Ninterval: %d\n',Ninterval);
fprintf('  Nduration: %d\n',Nduration);

% Set up the environment
addpath('../libsvm-mat-3.0-1/');
load dataset;

% Experiment for each dataset
for did = 1:length(D)
    % Create unique index of (time,person1,person2)
    persons = D(did).persons(:,1)';
    IDX = cell(length(persons),1);
    for i = 1:length(persons)
        time = D(did).observations(...
            D(did).observations(:,2)==persons(i) &...
            D(did).observations(:,7)==true,1);
        others = cell2mat(arrayfun(@(t)...
            D(did).observations(D(did).observations(:,1)==t &...
                                D(did).observations(:,2)~=persons(i) &...
                                D(did).observations(:,7)==true, 1:2),...
            time,'UniformOutput',false));
        if ~isempty(time) && ~isempty(others)
            IDX{i} = [others(:,1) ...
                      repmat(persons(i),[size(others,1) 1])...
                      others(:,2)];
        else
            IDX{i} = zeros(0,3);
        end
    end
    IDX = cat(1,IDX{:}); % (t,p1,p2)
    Pairs = unique(IDX(:,2:3),'rows');
    
    % Set seed points to sub-sample trajectories
    seeds = false(size(IDX,1),1);
    for i = 1:size(Pairs,1)
        ind = find(IDX(:,2)==Pairs(i,1)&IDX(:,3)==Pairs(i,2));
        seeds(ind(length(ind):-Ninterval:1)) = true;
    end
    seeds = find(seeds);
    
    % Compute feature representation for each sub-sampled trajectories
    F = zeros(length(seeds), grpFeature('len'));
    Y = zeros(length(seeds),1); % Labels
    for i = 1:length(seeds)
        S = IDX(seeds(i),:);  % Query seed index (t,p1,p2)
        X1 = D(did).observations(...            % Query person1's data
            D(did).observations(:,1)<=S(1) &... %   Past, and
            D(did).observations(:,2)==S(2),...  %   Person S(2)
            [1 3:6]);
        X2 = D(did).observations(...            % Query person2's data
            D(did).observations(:,1)<=S(1) &... %   Past, and
            D(did).observations(:,2)==S(3),...  %   Person S(3)
            [1 3:6]);
        X1 = X1(size(X1,1)-min(size(X1,1),Nduration)+1:end,:);
        X2 = X2(size(X2,1)-min(size(X2,1),Nduration)+1:end,:);
        F(i,:) = grpFeature(X1,X2); % Compute feature
        Y(i) = D(did).persons(D(did).persons(:,1)==S(2),4) ==...
               D(did).persons(D(did).persons(:,1)==S(3),4); % Keep truth
    end
    
    % Train and test
    C = cell(Nfold,1);
    fprintf('Dataset: %s\n',D(did).label);
    for i = 1:Nfold
%         ind = mod(1:length(Y),Nfold)==i-1;
        ind = ceil((1:length(Y))/length(Y)*Nfold)==i;
        % Train
        C{i} = svmtrain(Y(~ind),F(~ind,:),'-q');
        % Test
        [Yhat,A] = svmpredict(Y(ind),F(ind,:),C{i});
        fprintf(' Accuracy %f%% (%d/%d)',...
            100*nnz(Yhat==1&Y(ind)==1)/nnz(Y(ind)==1),...
            nnz(Yhat==1&Y(ind)==1),nnz(Y(ind)==1));
        
        % Apply reflectivity constraints
        I = IDX(seeds(ind),:);   % Set of pairs considered over time t
        time = unique(I(:,1))';
        for t = time
            ind2 = find(I(:,1)==t);
            PP = I(ind2,2:3);   % Set of pairs at t
            persons = unique([PP(:,1);PP(:,2)])';
            M = false(length(persons));
            for j = 1:size(PP,1)
                M(PP(j,1)==persons,PP(j,2)==persons) = Yhat(ind2(j));
            end
            M = M | M';
            M0 = false(length(persons));
            while ~all(M0==M)
                M0 = M;
                for j = 1:size(M,1)
                    M(j,:) = M(j,:) | any(M(M(j,:),:),1);
                    M(j,j) = false;
                end
            end
            % Fill in the Yhat
            y = false(length(ind2),1);
            for j = 1:size(PP,1)
                y(j) = M(PP(j,1)==persons,PP(j,2)==persons);
            end
            if any(Yhat(ind2) & ~y), keyboard; end
            Yhat(ind2) = y;
        end
        fprintf(', reflectivity: %f%% (%d/%d)\n',...
            100*nnz(Yhat==1&Y(ind)==1)/nnz(Y(ind)==1),...
            nnz(Yhat==1&Y(ind)==1),nnz(Y(ind)==1));
    end
end

end
