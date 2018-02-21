function [ err, phat, vhat ] = ewapError( T, Obj, params )
%EWAPERROR computes error in prediction given data
%
%   [err] = ewapobjective(T,params)
%
% Input:
%   T: Table(dataset,t,id,px,py,vx,vy,pnextx,pnexty,pdestx,pdesty,u)
%   Others: cell array of associated tuples (px,py,vx,vy,groups) of other
%           pedestrians
%   params: 1-by-6 parameter vector
% Output:
%   err : N-by-1 vector of errors for each row
%   phat: N-by-2 row vectors of predicted position
%   vhat: N-by-2 row vectors of predicted velocity

%% Arrange data structure
D = T(T(:,13)==1,:); % Pick valid observations
% Preallocate relations
Others = cell(size(D,1),1);
for i = 1:size(D,1)
    Others{i} = T(T(:,1) == D(i,1) &... % same dataset
                  T(:,2) == D(i,2) &... % same time
                  T(:,3) ~= D(i,3), ... % different id,...
                  [3 4 5 6 7]);         % query [pid,px,py,vx,vy]
end

%% Compute prediction error
err = zeros(size(D,1),2);
phat = zeros(size(D,1),2);
vhat = zeros(size(D,1),2);
parfor i = 1:size(D,1)
    t = D(i,:);                         % Query tuple of data i
    o = Obj; o = o(o(:,1)==t(1),2:3);   % Query obstacles
    % Compute optimal velocity choice
    vhat(i,:) = fminunc(@(x) ewapEnergy(x,...
                            [t([4 5]); Others{i}(:,[2 3]); o],...  % p
                            [t([6 7]); Others{i}(:,[4 5]); zeros(size(o))],...  % v
                            t(12),t([10 11]),...                % ui,zi
                            params...                          % params
                            ),...
                        t([6 7]),...                        % init value
                        optimset('GradObj','on',...
                            ...'DerivativeCheck','on',...
                            'LargeScale','off',...
                            'Display','off'...
                            ));
    % Predict the next position
    phat(i,:) = t([4 5])+0.4*(params(6)*t([6 7])+(1-params(6))*vhat(i,:));
    % Compute the difference between truth and prediction in position
    err(i,:) = t([8 9]) - phat(i,:);
end

end
