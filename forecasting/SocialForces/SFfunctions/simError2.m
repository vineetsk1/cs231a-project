function [ e ] = simError2( S, method, params, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Options
DEBUG = false;
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'DEBUG'), DEBUG = varargin{i+1}; end
end

% Drop image processing
for i = 1:length(S)
    S(i).video = [];
    S(i).H = [];
end

% Main loop
e = cell(length(S),1);
if DEBUG
    for i = 1:length(S)
        e{i} = simulate(S(i),method,params,...
            'AppMethod','none','PredictSpeed',false,varargin{:});
    end
else
    parfor i = 1:length(S)
        e{i} = simulate(S(i),method,params,...
            'AppMethod','none','PredictSpeed',false,varargin{:});
    end
end
e = cell2mat(e);   % Assuming only 1 dataset is contained

end

