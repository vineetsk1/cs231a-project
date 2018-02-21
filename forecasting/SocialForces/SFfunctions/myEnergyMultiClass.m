function [E,dE] = myEnergyMultiClass( vhat, p, v, ui, zi, params, groups, class )
%EWAPENERGY energy function of individual pedestrians
%
%   [E] = ewapEnergy(vhat,p,v,ui,zi,params)
%   [E,dE] = ewapEnergy(vhat,p,v,ui,zi,params)
%
% Input:
%   vhat: 1-by-2 vector of velocity in the next time step
%   p: n-by-2 row vectors of pedestrian positions (xi, yi)
%      p(1,:) is of the subject of interest
%      p(2:end,:) is of other pedestrians
%   v: n-by-2 row vectors of pedestrian velocities (dxi/dt, dyi/dt)
%      v(1,:) is of the subject of interest
%      v(2:end,:) is of other pedestrians
%   ui: comfortable speed for pedestrian i
%   zi: 1-by-2 row vector of destination for pedestrian i
%   params: parameters of the energy function
%   groups: indicator vector of attracting pedestrians
% Output:
%   E: energy of the input
%   dE: gradient of energy to the input

cl = class;  % agent' class

vhat(isnan(vhat)) = 0; % Treat NaN as zero

% Magnitude and normalization of vhat
[vq,vr] = cart2pol(vhat(1),vhat(2));

% Free flow term 
S = (ui - vr)^2;

% Destination term
z = cart2pol(zi(1)-p(1,1),zi(2)-p(1,2)); % normalized dir to dest
if vr == 0, vq = z; end % if dir of vhat is undefined, set it to goal dir
D = - [cos(z) sin(z)] * [cos(vq);sin(vq)];

% Interaction term
I = 0;
if size(p,1) > 1
    k = repmat(p(1,:),[size(p,1)-1 1]) - p(2:end,:);
    q = repmat(vhat,[size(v,1)-1 1]) - v(2:end,:);
    [kphi,kr] = cart2pol(k(:,1),k(:,2));
    [qphi,qr] = cart2pol(q(:,1),q(:,2));
    kq = sum(k.*q,2);
    
    % Energy Eij
    dstar = (k - (kq./(qr.^2)) * ones(1,2) .* q);     % k - k.q/|q|^2 q
    dstar(isnan(dstar)) = inf;
    eij = exp(-0.5*(sum(dstar.^2,2))/params{cl}(1)^2);
    
    % Coefficients wd and wf
    phi = kphi-atan2(v(1,2),v(1,1));
    wd = exp(-0.5*kr.^2/params{cl}(2)^2);
    wf = (0.5*(1+cos(phi))).^params{cl}(3);
    wf(2*abs(phi)>pi) = 0;
    
    I = sum(wd.*wf.*eij);
end


% Attraction term
A1 = 0;
if size(p,1) > 1 && any(groups(2:end))
    % Get the direction and distance to the group centroid
    % Weight to guarantee that they have similar current velocity
    wa = cos(atan2(v(1,2),v(1,1)) -...
        atan2(v([false;groups(2:end)],2),v([false;groups(2:end)],1)));
%     wa(wa<0) = 0; % isnan
    % Attraction potential
    A1 = sum(wa.*cos(kphi(groups(2:end))-vq),1);
end

% Attraction term 2 (grouping)
A2 = 0;
if size(p,1) > 1 && any(groups(2:end))
    A2 = sum((vhat - mean(v(groups,:),1)).^2,2);
end

% Damping term
V = sum((vhat - v(1,:)).^2);

% Summation of individual terms
E = I + params{cl}(4) * S + params{cl}(5) * D +...
    params{cl}(6) * A1 + params{cl}(7) * A2 +...
    params{cl}(8) * V;

% Compute analytical gradient if necessary
% might have a bug at extrema
if nargout > 1
    % Free flow term
    dS = -2*(ui - vr)*[cos(vq) sin(vq)];
    
    % Destination term
    dD = -[cos(z) sin(z)] * ...
         [[sin(vq);-cos(vq)]*sin(vq) [-sin(vq);cos(vq)]*cos(vq)]/vr;
    dD(isnan(dD)|isinf(dD)) = 0;
    
    % Interaction term
    dI = [0 0];
    if size(p,1) > 1
        dxx = (-k(:,1).*q(:,1)+kq.*(2*(q(:,1).^2)./(qr.^2)-1))./(qr.^2);
        dyx = (-k(:,1).*q(:,2)+kq.*(2*(q(:,1).*q(:,2))./(qr.^2)))./(qr.^2);
        dxy = (-k(:,2).*q(:,1)+kq.*(2*(q(:,2).*q(:,1))./(qr.^2)))./(qr.^2);
        dyy = (-k(:,2).*q(:,2)+kq.*(2*(q(:,2).^2)./(qr.^2)-1))./(qr.^2);
        dI = - repmat(wd.*wf.*eij,[1 2]) / (params{cl}(2)^2) .* ...
            [dstar(:,1).*dxx+dstar(:,2).*dyx...
             dstar(:,1).*dxy+dstar(:,2).*dyy];
        dI(isnan(dI)|isinf(dI)) = 0;
    end
    
    
    % Attraction term
    dA1 = [0 0];
    if size(p,1) > 1 && any(groups(2:end))
        dA1 = [cos(kphi(groups(2:end))).*sin(vq).^2-sin(kphi(groups(2:end))).*cos(vq).*sin(vq)...
              -sin(kphi(groups(2:end))).*cos(vq).*sin(vq)+sin(kphi(groups(2:end))).*cos(vq).^2 ]...
              ./ [kr(groups(2:end)) kr(groups(2:end))] .* [wa wa];
    end
    
    % Attraction term 2 (grouping)
    dA2 = [0 0];
    if size(p,1) > 1 && any(groups(2:end))
        dA2 = 2*(vhat - mean(v(groups,:),1));
    end
    
    % Damping term
    dV = 2*(vhat - v(1,:));
    
    % Summation of individual terms
    dE = sum(dI,1) + params{cl}(4) * dS + params{cl}(5) * dD +...
        params{cl}(6) * sum(dA1,1) + params{cl}(7) * dA2 +...
        params{cl}(8) * dV;
end

end
