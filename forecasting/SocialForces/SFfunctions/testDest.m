function [] = testDest()
%TESTDEST evaluates quality of the destination prediction

% Config
Ninterval = 1;  % Subsample seed interval
Nduration = 2;  % Subsample duration
XBinSize = 5;   % Discretization in Px
YBinSize = 5;   % Discretization in PY
RBinSize = 5;   % Discretization in magnitude(V)
TBinSize = 9;   % Discretization in angle(V)
GridSize = 1;   % Grid size in time, must be smaller than Nduration
Nfold = 3;      % Cross validation

% Set up the environment
addpath('../libsvm-mat-3.0-1/');
load dataset;

% Convert velocity from cart to polar coord
for did = 1:length(D)
    X = D(did).observations(:,5:6);
    [D(did).observations(:,6),D(did).observations(:,5)] =...
        cart2pol(X(:,1),X(:,2));
end

% Experiment for each dataset
for did = 1:length(D)
    % Set up feature discritization params
    XBins = [-inf linspace(min(D(did).observations(:,3)),...
                     max(D(did).observations(:,3)),XBinSize) inf];
    YBins = [-inf linspace(min(D(did).observations(:,4)),...
                     max(D(did).observations(:,4)),YBinSize) inf];
    RBins = [linspace(min(D(did).observations(:,5)),...
                     max(D(did).observations(:,5)),RBinSize) inf];
    TBins = linspace(-pi,pi,TBinSize);
    
    % Set seed points to sub-sample trajectories
    persons = D(did).persons(:,1)';
    seeds = false(size(D(did).observations,1),1);
    for i = 1:length(persons)
        ind = find(D(did).observations(:,2)==persons(i) &...
                   D(did).observations(:,7)==1);
        seeds(ind(length(ind):-Ninterval:1)) = true;
    end
    seeds = find(seeds);
    
    % Compute feature representation for each sub-sampled trajectories
    F = zeros(length(seeds),...
              trajFeature([],[],[],[],...
                         'XBins',XBins,'YBins',YBins,...
                         'RhoBins',RBins,'ThetaBins',TBins,...
                         'TimeGridSize',GridSize,'len',1));
    Y = zeros(length(seeds),1); % Labels
    for i = 1:length(seeds)
        S = D(did).observations(seeds(i),1:2);  % Query seed tuple
        X = D(did).observations(...             % Query this person's data
            D(did).observations(:,1)<=S(1) &... %   Past, and
            D(did).observations(:,2)==S(2),:);  %   The same person
        X = X(size(X,1)-min(size(X,1),Nduration)+1:end,:);   % Drop excess
        F(i,:) = trajFeature(X(:,3),X(:,4),X(:,5),X(:,6),... % Compute f
                             'XBins',XBins,'YBins',YBins,...
                             'RhoBins',RBins,'ThetaBins',TBins,...
                             'TimeGridSize',GridSize);
        Y(i) = D(did).persons(D(did).persons(:,1)==S(2),2); % Keep label
    end
    
    % Train and test
    C = cell(Nfold,1);
    fprintf('Dataset: %s\n',D(did).label);
    for i = 1:Nfold
        ind = mod(1:length(Y),Nfold)==i-1;
%         ind = ceil((1:length(Y))/length(Y)*Nfold)==i;
        % Train
        C{i} = svmtrain(Y(~ind),F(~ind,:),'-q');
        % Test
        [Yhat,A] = svmpredict(Y(ind),F(ind,:),C{i});
        fprintf(' Accuracy %f%% (%d/%d)\n',A(1),nnz(Yhat==Y(ind)),nnz(ind));
    end
end

end
