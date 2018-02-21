function [ H ] = trajFeature( X, Y, rho, theta, varargin )
%TRAJFEATURE computes feature representation of a trajectory

% Set up options
XBins = linspace(0,1,5);
YBins = linspace(0,1,5);
RhoBins = linspace(0,1,5);
ThetaBins = linspace(-pi,pi,9);
gridSize = 1;
returnLen = false;
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'XBins'), XBins = varargin{i+1}; end
    if strcmp(varargin{i},'YBins'), YBins = varargin{i+1}; end
    if strcmp(varargin{i},'RhoBins'), RhoBins = varargin{i+1}; end
    if strcmp(varargin{i},'ThetaBins'), ThetaBins = varargin{i+1}; end
    if strcmp(varargin{i},'TimeGridSize'),gridSize = varargin{i+1};end
    if strcmp(varargin{i},'len'),       returnLen = varargin{i+1}; end
end

% Return length of the feature if input is 'len'
if returnLen
    H = gridSize*(length([XBins YBins RhoBins ThetaBins]));
    return;
end

% Check error
X = X(:)'; Y = Y(:)'; rho = rho(:)'; theta = theta(:)';
if length(X)~=length(Y)||length(X)~=length(rho)||length(X)~=length(theta)
    error('Argument format invalid');
end

% Compute feature
H = cell(1,gridSize);
for i = 1:gridSize
    ind = round(linspace(1,gridSize,length(X)))==i;
    if ~any(ind) || isempty(X)
        H{i} = zeros(1,length([XBins YBins RhoBins ThetaBins]));
    else
        H{i} = [histc(X(ind),XBins)     histc(Y(ind),YBins)...
                histc(rho(ind),RhoBins) histc(theta(ind),ThetaBins)]...
                /nnz(ind);
    end
end
H = [H{:}];

end
