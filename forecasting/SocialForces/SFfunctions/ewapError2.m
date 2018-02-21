function [ err ] = ewapError2( T, Others, Obj, params, varargin )
%EWAPERROR2 computes norm of the partial derivative of individual energy
%
%   [err] = ewapError2(T,theta)
%
% Input:
%   T: Table(dataset,t,id,px,py,vx,vy,pnextx,pnexty,pdestx,pdesty,u)
%   Others: cell array of associated tuples (px,py,vx,vy,groups) of other
%           pedestrians
%   theta: 1-by-6 parameter vector
% Output:
%   err : N-by-1 vector of errors for each row

regularization = zeros(0,length(params));
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Regularization'), regularization = varargin{i+1}; end
end

% Compute prediction error
err = zeros(size(T,1),2);
parfor i = 1:size(T,1)
    % Query tuple of data i
    t = T(i,:);
    theta = params;
    o = Obj;
    o = o(o(:,1)==t(1),2:3);
    % Get vhat
    vhat = (2.5*(t([8 9])-t([4 5]))-theta(6)*t([6 7]))/(1-theta(6));
    % Compute optimal velocity choice
    [E,dE] = ewapEnergy(vhat,...
                        [t([4 5]); Others{i}(:,[2 3]);o],...  % p
                        [t([6 7]); Others{i}(:,[4 5]);zeros(size(o))],...  % v
                        t(12),t([10 11]),...                % ui,zi
                        theta);
    % Compute the difference between truth and prediction in position
    err(i,:) = dE;
end

% With reguralization
r = zeros(size(regularization,1),1);
for i = 1:size(regularization,1)
    R = regularization(i,:);
    % Treat negative term to be inverse
    if all(R>=0)
        r(i) = size(T,1)*params(R>0)*diag(R(R>0))*params(R>0)';
    else
        r(i) = size(T,1)/(params(R<0)*diag(-R(R<0))*params(R<0)');
    end
end
err = [err(:);sqrt(r)];

end