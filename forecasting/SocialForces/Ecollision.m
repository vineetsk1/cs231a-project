function I = Ecollision(sr,params,vhat,p,v)

vhat(isnan(vhat)) = 0; % Treat NaN as zero

% Interaction term
I = 0;
if size(p,1) > 1
    k = repmat(p(1,:),[size(p,1)-1 1]) - p(2:end,:);
    q = repmat(vhat,[size(v,1)-1 1]) - v(2:end,:);
    [kphi,kr] = cart2pol(k(:,1),k(:,2));
    [~,qr] = cart2pol(q(:,1),q(:,2));
    kq = sum(k.*q,2);
    
    % Energy Eij
    dstar = (k - (kq./(qr.^2)) * ones(1,2) .* q);     % k - k.q/|q|^2 q
    dstar(isnan(dstar)) = inf;
    eij = exp(-0.5*(sum(dstar.^2,2))/sr^2);
    
    % Coefficients wd and wf
    phi = kphi-atan2(v(1,2),v(1,1))-pi;
    wd = exp(-0.5*kr.^2/params(1)^2);
    wf = (0.5*(1+cos(phi))).^params(2);
    %wf(2*abs(phi)>pi) = 0;
    
    I = sum(wd.*wf.*eij);
end
