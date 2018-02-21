function [ X ] = grpFeature( Ti,Tj )
%GRPFEATURE Summary of this function goes here
%   Detailed explanation goes here

%FEATURE computes feature
%
% Ti: trajectory(time,px,py,vx,vy)
% Tj: trajectory(time,px,py,vx,vy)

nbins = 9;  % number of bins
maxD = 5;   % maximum distance to discretize (meter)
dbins = [linspace(0,1,nbins) inf];
abins = pi*linspace(-1,1,nbins);

% If input is string, return the length of feature vector
if ischar(Ti), X = 4*nbins+1; return; end

X = zeros(nbins,4);

% Get time-aligned trajectories
frames = intersect(Ti(:,1),Tj(:,1)); % Find common frame id
Xi = Ti(arrayfun(@(x) any(x==frames),Ti(:,1)),2:5);
Xj = Tj(arrayfun(@(x) any(x==frames),Tj(:,1)),2:5);
dP = Xi(:,1:2)-Xj(:,1:2);

% Distance histogram
h = histc(sqrt(sum(dP.^2,2)),maxD*dbins)/length(frames);
X(:,1) = h(1:end-1);
% Angle between dP and Velocity
phi = atan2(dP(:,2),dP(:,1))-atan2(Xi(:,4),Xi(:,3));
phi(phi>pi) = phi(phi>pi) - 2*pi;
phi(phi<-pi) = phi(phi<-pi) + 2*pi;
X(:,2) = histc(abs(phi),abins)/length(frames);
% Difference in velocity magnitude
h = histc(abs(sqrt(sum((Xi(:,3:4)).^2,2))-...
              sqrt(sum((Xj(:,3:4)).^2,2))),dbins)/length(frames);
X(:,3) = h(1:end-1);
% Angle between velocity direction
phi = atan2(Xj(:,4),Xj(:,3))-atan2(Xi(:,4),Xi(:,3));
phi(phi>pi) = phi(phi>pi) - 2*pi;
phi(phi<-pi) = phi(phi<-pi) + 2*pi;
X(:,4) = histc(abs(phi),abins)/length(frames);

% Append overlap ratio to the end
X = [X(:)' length(frames)/length(union(Ti(:,1),Tj(:,1)))];

end

