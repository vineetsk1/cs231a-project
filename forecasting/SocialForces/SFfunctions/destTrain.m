function [ C, R ] = destTrain( Obs, Sims, varargin )
%DESTTRAIN trains destination classifier
%
% Input:
%   Obs:  table(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Sims: table(simulation,dataset,person,start,duration)
% Output:
%   C:    cell array of SVM struct
%   R:    table(dataset,fold,truth,prediction)

% Config
XBinSize = 5;   % Discretization in Px
YBinSize = 5;   % Discretization in PY
RBinSize = 5;   % Discretization in magnitude(V)
TBinSize = 9;   % Discretization in angle(V)
Nfold = 1;      % Internal cross validation
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Nfold'), Nfold = varargin{i+1}; end
end

% Experiment for each dataset
datasets = unique(Sims(:,2))';
C = cell(max(datasets),1);
R = cell(max(datasets),1);
for did = datasets
    % Query observations
    s = Sims(Sims(:,2)==did,:);                        % simulation index
    o = Obs(arrayfun(@(x) any(x==s(:,1)),Obs(:,1)),:); % simulation data
    
    % Set up feature discritization params
    Rho = sqrt(sum(o(:,5:6).^2,2));   % Speed
    C{did}.XBins = [-inf linspace(min(o(:,3)),max(o(:,3)),XBinSize) inf];
    C{did}.YBins = [-inf linspace(min(o(:,4)),max(o(:,4)),YBinSize) inf];
    C{did}.RBins = [-inf linspace(min(Rho),max(Rho),RBinSize) inf];
    C{did}.TBins = linspace(-pi,pi,TBinSize);
    
    % Compute feature representation for each trajectories
    F = zeros(size(s,1),...
              destFeature([],[],[],[],...
                         'XBins',C{did}.XBins,...
                         'YBins',C{did}.YBins,...
                         'RhoBins',C{did}.RBins,...
                         'ThetaBins',C{did}.TBins,...
                         'len',true));
    Y = zeros(size(s,1),1); % Labels
    for i = 1:size(s,1)
        X = o(o(:,1)==s(i,1) & o(:,3)==s(i,3),:);
        F(i,:) = destFeature(X(:,4),X(:,5),X(:,6),X(:,7),...
                             'XBins',C{did}.XBins,...
                             'YBins',C{did}.YBins,...
                             'RhoBins',C{did}.RBins,...
                             'ThetaBins',C{did}.TBins);
        Y(i) = X(end,8);    % Destination label at the end
    end
    
    % Train and test
    fprintf('  Dataset: %d\n',did);
    c = cell(Nfold,1);
    A = zeros(Nfold,1);
    Yhat = Y;
    I = zeros(size(Y));
    for i = 1:Nfold
        ind = mod(1:length(Y),Nfold)==i-1;
%         ind = ceil((1:length(Y))/length(Y)*Nfold)==i;
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
    C{did}.C = c{ind};
    
    R{did} = [did*ones(size(I,1),1) I Y Yhat];
end
R = cell2mat(R);    % Report the result

end
