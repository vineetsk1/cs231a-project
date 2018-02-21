function [ C, R ] = grpTrain2( Obs, Sims, varargin )
% Function to compute features for all possible pairs of tuples
%
% Input:
%   Obs:  table(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Sims: table(simulation,dataset,person,start,duration)
% Output:
%   C:    SVM struct
%   R:    table(fold,truth,prediction)

% Config
Nfold = 1;      % Internal cross validation
maxDist = 6.0;  % Maximum distance to consider grouping in meter
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Nfold'), Nfold = varargin{i+1}; end
    if strcmp(varargin{i},'MaxDist'), maxDist = varargin{i+1}; end
end

% Compute feature representation for each pair of trajectories
F = cell(size(Sims,1),1); % Features
Y = cell(size(Sims,1),1); % Labels
for i = 1:size(Sims,1)
    X = Obs(Obs(:,1)==Sims(i,1),:);
    pid = Sims(i,3);
    
    % Drop too distant data
    ind = true(size(X,1),1);
    time = X(X(:,3)==pid,2);
    for t = time'
        ind2 = find(X(:,2)==t);
        x = X(ind2,:); % data at time t
        d = sqrt(sum((x(:,4:5)-repmat(x(x(:,3)==pid,4:5),[size(x,1) 1])).^2,2));
        ind(ind2(d>maxDist)) = false;
    end
    X = X(ind,:);
    
    % Find other pedestrians
    others = unique(X(X(:,3)~=pid,3))';
    F{i} = zeros(length(others),grpFeature('len'));
    Y{i} = zeros(length(others),1);
    for k = 1:length(others)
        F{i}(k,:) = grpFeature(X(X(:,3)==pid,[2 4:7]),...
                               X(X(:,3)==others(k),[2 4:7]));
        Y{i}(k) = X(find(X(:,3)==pid,1,'first'),10)==...
                  X(find(X(:,3)==others(k),1,'first'),10);
    end
end
F = cat(1,F{:});
Y = cat(1,Y{:});

% Train and test
c = cell(Nfold,1);
A = zeros(Nfold,1);
Yhat = Y;
I = zeros(size(Y));
for i = 1:Nfold
    ind = mod(1:length(Y),Nfold)==i-1;
%     ind = ceil((1:length(Y))/length(Y)*Nfold)==i;
    % Train
    if Nfold==1
        c{i} = svmtrain(Y(ind),F(ind,:),'-q');
    else
        c{i} = svmtrain(Y(~ind),F(~ind,:),'-q');
    end
    % Test
    [Yhat(ind),a] = svmpredict(Y(ind),F(ind,:),c{i});
    A(i) = a(1);
    fprintf('    Fold %d: Accuracy %f%% (%d/%d)\n',...
            i,a(1),nnz(Yhat(ind)==Y(ind)),nnz(ind));
    % Keep the number of fold
    I(ind) = i;
end
[tmp,ind] = max(A);
C.C = c{ind};

R = [I Y Yhat];

end

