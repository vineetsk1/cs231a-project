
function [ err, Pred, Grp ] = simErrorM( Obs, Sims, method, varargin )
%SIMERROR multiple behavioral simulation and error evaluation
%
% Input:
%   Obs(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Sims(simulation,dataset,person,start,duration)
%   method: either of 'lin', 'ewap', 'attr'
%   options:
% Output:
%   err: error array for each simulation step
%   Pred: result table for each simulation step
%         (simulation,time,person,px,py,vx,vy,dest,speed
%   Grp: predicted group relation for the subject of simulation
%         (simulation,time,pid)
%        That is, pid in the Sims table has relation to pid in the Grp

% Options

Cd = [];
Cg = [];
Npast = 0;      % Past steps to use in destination/group prediction
params = [];
Dest = [];
Obst = [];
maxDist = 6.0;  % Maximum distance to consider grouping
dt = 0.4;       % Timestep
vid = [];
H = [];
winSize = [48 24];
tmplSize = [32 16];
anchorPos = [-36 0];
sa = 1;
sb = 1000;
appMethod = 'mix';
predictSpeed = true;
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'DestClassifier'), Cd = varargin{i+1}; end
    if strcmp(varargin{i},'GroupClassifier'), Cg = varargin{i+1}; end
    if strcmp(varargin{i},'Past'), Npast = varargin{i+1}; end
    if strcmp(varargin{i},'Params'), params = varargin{i+1}; end
    if strcmp(varargin{i},'Dest'), Dest = varargin{i+1}; end
    if strcmp(varargin{i},'Obst'), Obst = varargin{i+1}; end
    if strcmp(varargin{i},'MaxDist'), maxDist = varargin{i+1}; end
    if strcmp(varargin{i},'Video'), vid = varargin{i+1}; end
    if strcmp(varargin{i},'H'), H = varargin{i+1}; end
    if strcmp(varargin{i},'AppMethod'), appMethod = varargin{i+1}; end
    if strcmp(varargin{i},'WinSize'), winSize = varargin{i+1}; end
    if strcmp(varargin{i},'TmplSize'), tmplSize = varargin{i+1}; end
    if strcmp(varargin{i},'AnchorPos'), anchorPos = varargin{i+1}; end
    if strcmp(varargin{i},'SigmaA'), sa = varargin{i+1}; end
    if strcmp(varargin{i},'SigmaB'), sb = varargin{i+1}; end
    if strcmp(varargin{i},'PredictSpeed'), predictSpeed = varargin{i+1}; end
end

%% Check args
if ~any(cellfun(@(x)isempty(x), strfind({'lin','ewap','attr'},method)))
    error('Invalid method');
end
if strcmp(method,'ewap') || strcmp(method,'attr')
    if isempty(params) || isempty(Dest), error('Missing arguments'); end
end

% Convert Obs and Sims to cell array format
simulations = unique(Sims(:,1))';
D = cell(length(simulations),1);
for i = 1:length(simulations)
    D{i} = Obs(Obs(:,1)==simulations(i),:);
end
S = cell(length(simulations),1);
for i = 1:length(simulations)
    S{i} = Sims(Sims(:,1)==simulations(i),:);
end

%% Initialization
err = cell(length(simulations),1);
Pred = cell(length(simulations),1);
Grp = cell(length(simulations),1);
% Do simulations
for i = 1:size(simulations,1)
    % Query simulation info
    s = S{i};
    time = unique(D{i}(:,2))';
    time = time(min(s(:,4))<=time &...
                time<=max(arrayfun(@(j)...
                    time(min(find(time==s(j,4))+s(j,5),length(time))),...
                    1:size(s,1))));
    % Set temporary variables
    theta = params;                                     % parameters
    pDest = Dest; pdest = pDest(pDest(:,1)==s(1,2),2:3);  % dest(x,y)
    pObst = Obst; o = pObst(pObst(:,1)==s(1,2),2:3);      % obst(x,y)
    %% Initialize variables for simulation
    % Query for each person the future information during sim
    Osim = cell2mat(arrayfun(@(j) ...
        D{i}(D{i}(:,2)>=s(j,4) &...      % from start time
             D{i}(:,2)<=time(find(time==s(j,4))+s(j,5)) &... % to the end
             D{i}(:,3)==s(j,3),:),...    % of the person
        (1:size(s,1))','UniformOutput',false));
    
    g = cell(length(time)-1,1);          % group table skelton
    
    % Initialize background image model
    if ~isempty(vid) && ~isempty(H)
        bg = zeros(vid.Height,vid.Width,3,'single'); % background image
        past = unique(D{i}(:,2))';
        past = past(past<=time(1));
        % Accumulate background info up until time(1)
        for j = 1:length(past)
            [img,bg] = cvbgsubtract2(read(vid,past(j)),bg,j,25);
        end
        Tpast = length(past);
    end
%     img = read(vid,time(1));
    
    %% Simulate the behavior of person pid
    for j = 1:length(time)-1
        %% Predict comfortable speed = running average
        if predictSpeed
            persons = Osim(Osim(:,2)==time(j),3)';
            for pid = persons
                Osim(Osim(:,2)==time(j+1) & Osim(:,3)==pid,9) =...
                     mean(sqrt(sum(Osim(Osim(:,2)<=time(j)&...
                                        Osim(:,3)==pid,6:7).^2,2)));
            end
        end
        
        %% Predict dest
        % Output: dhat(j+1)
        % TODO: what about prediction at time(end)?
        if ~isempty(Cd)
            persons = Osim(Osim(:,2)==time(j),3)';
            for pid = persons
                % Query (p,v)
                X = [D{i}(D{i}(:,2)>=time(max(1,j-Npast)) &...
                          D{i}(:,2)<s(s(:,3)==pid,4) &...
                          D{i}(:,3)==pid,4:7);...
                     Osim(Osim(:,2)>=time(max(1,j-Npast)) &...
                             Osim(:,2)<=time(j) &...
                             Osim(:,3)==pid,4:7)];
                % Compute features
                F = destFeature(X(:,1),X(:,2),X(:,3),X(:,4),...
                                'XBins',Cd{s(1,2)}.XBins,...
                                'YBins',Cd{s(1,2)}.YBins,...
                                'RhoBins',Cd{s(1,2)}.RBins,...
                                'ThetaBins',Cd{s(1,2)}.TBins);
                % Predict labels
                Osim(Osim(:,2)==time(j)&...
                        Osim(:,3)==pid,8) = svmpredict(1,F,Cd{s(1,2)}.C);
            end
        end
        
        %% Predict group
        % Output: g{j}
        x = grpTable(Osim(Osim(:,2)==time(j),:),maxDist);
        % TODO: How do we deal with people not to simulate?
        if isempty(x)
            g{j} = zeros(0,7);
        else
            g{j} = [repmat(i,[size(x,1) 1]) x];
            if ~isempty(Cg)
                for k = find(g{j}(:,7)'==1)
                    % Query (p,v) of person g{j}(k,3)
                    X1 = [D{i}(D{i}(:,2)>=time(max(1,j-Npast)) &...
                               D{i}(:,2)<s(s(:,3)==g{j}(k,3),4) &...
                               D{i}(:,3)==g{j}(k,3),[2 4:7]);...
                          Osim(Osim(:,2)>=time(max(1,j-Npast)) &...
                                  Osim(:,2)<=time(j) &...
                                  Osim(:,3)==g{j}(k,3),[2 4:7])];
                    % Query (p,v) of person g{j}(k,4)
                    X2 = [D{i}(D{i}(:,2)>=time(max(1,j-Npast)) &...
                               D{i}(:,2)<s(s(:,3)==g{j}(k,4),4) &...
                               D{i}(:,3)==g{j}(k,4),[2 4:7]);...
                          Osim(Osim(:,2)>=time(max(1,j-Npast)) &...
                                  Osim(:,2)<=time(j) &...
                                  Osim(:,3)==g{j}(k,4),[2 4:7])];
                    % Predict label
                    g{j}(k,6) = logical(svmpredict(1,grpFeature(X1,X2),Cg.C));
                end
            end
        end

        %% Predict velocity and position
        
        % Get behavioral cue
        curr = Osim(:,2)==time(j);             % index
        next = Osim(:,2)==time(j+1);           % index
        % (person,phat,vhat,p): persons necessary to be predicted
        subject = Osim(curr,3);
        subject = subject(arrayfun(@(pid) any(Osim(next,3)==pid), subject));
        Pb = [subject zeros(length(subject),6)];
        for pid = Pb(:,1)'
            subject = Osim(:,3)==pid;          % index
            others = Osim(:,3)~=pid;           % index
            if ~any(next & subject), continue; end
            groups = logical(arrayfun(@(qid)...
                g{j}(g{j}(:,3)==pid&g{j}(:,4)==qid,6),...
                Osim(curr & others,3)));
            [phat,vhat] = behaviorPredict(method,...
                [Osim(curr & subject,4:5);o;Osim(curr & others,4:5)],... % position
                [Osim(curr & subject,6:7);zeros(size(o));Osim(curr & others,6:7)],...
                Osim(curr & subject,9),...              % ui
                pdest(Osim(curr & subject,8),:),...     % zi
                theta,...                               % params
                [true;false(size(o,1),1);groups],... % groups
                dt);
            % Position
            Pb(Pb(:,1)==pid,2:7) = [phat vhat Osim(curr & subject,4:5)];
        end
        % TODO: how to avoid tracker collision
        
        % Fix it with the appearance observation if available
        if ~isempty(vid) && ~isempty(H) && ~isempty(Pb)
            img0 = img; % Keep previous frame
            [img,bg] = cvbgsubtract2(read(vid,time(j+1)),bg,j+Tpast,25);
%             img = read(vid,time(j+1));
            [phat,vhat] = appearanceFeedback(appMethod,...
                Pb(:,6:7),Pb(:,2:3),img0,img,H,...
                'WinSize',winSize,'TmplSize',tmplSize,'AnchorPos',anchorPos,...
                'SigmaA',sa,'SigmaB',sb,'TimeStep',dt);
            Pb(:,2:5) = [phat vhat];

% DEBUG
fprintf('Frame %d\n',time(j));
disp(Pb);
imshow(img);
pcurr = [Pb(:,6:7) ones(size(Pb,1),1)] / H';
pcurr = [pcurr(:,1)./pcurr(:,3) pcurr(:,2)./pcurr(:,3)];
phat = [Pb(:,2:3) ones(size(Pb,1),1)] / H';
phat = [phat(:,1)./phat(:,3) phat(:,2)./phat(:,3)];
hold on;
plot(pcurr(:,2),pcurr(:,1),'wo');
plot(phat(:,2),phat(:,1),'w*');
plot([phat(:,2) pcurr(:,2)]',[phat(:,1) pcurr(:,1)]','w-');
text(phat(:,2)+5,phat(:,1)+10,cellstr(num2str(Pb(:,1))),'Color','w');
for k = Pb(:,1)'
    rectangle('Position',[phat(Pb(:,1)==k,2)-tmplSize(2)+anchorPos(2)...
                          phat(Pb(:,1)==k,1)-tmplSize(1)+anchorPos(1)...
                          tmplSize(2)*2 tmplSize(1)*2],...
              'EdgeColor','w');
    for l = g{j}(g{j}(:,3)==k&g{j}(:,6)==1,4)'
        plot([pcurr(Pb(:,1)==k,2) pcurr(Pb(:,1)==l,2)],...
             [pcurr(Pb(:,1)==k,1) pcurr(Pb(:,1)==l,1)],'g--');
    end
end
text(5,10,sprintf('%s  frame=%d',method,time(j)),'Color','w');
hold off;
drawnow; %pause(.5);

        end
        
        % Update position and velocity
        for pid = Pb(:,1)'
            Osim(Osim(:,2)==time(j+1) & Osim(:,3)==pid,4:7) =...
                Pb(Pb(:,1)==pid,2:5);
        end
    end
    
    %% Compute euclid distance
    err{i} = arrayfun(@(x) sqrt(sum((D{i}(...
        D{i}(:,2)==Osim(x,2) & D{i}(:,3)==Osim(x,3),4:5) -...
        Osim(x,4:5)).^2,2)), (1:size(Osim,1))');
    Pred{i} = Osim;
    Grp{i} = cat(1,g{:});
end

% Reshape output
err = cat(1,err{:});
Pred = cat(1,Pred{:});
Grp = cat(1,Grp{:});

end

%% SUBROUTINES

%% Group relation table
function [ groups ] = grpTable( X, maxDist )
%GRPTABLE constructs group table from tuples
% table(time,pid,pid,Y,Yhat,flag)

persons = X(:,3)';
Mt = false(length(persons));    % indicator matrix
Dt = zeros(length(persons));    % distance matrix
for k = 1:length(persons)
    Mt(k,:) = arrayfun(@(pid) X(X(:,3)==pid,10)==...
                              X(X(:,3)==persons(k),10),persons);
    Dt(k,:) = arrayfun(@(pid)...
        sqrt(sum((X(X(:,3)==pid,4:5)-X(X(:,3)==persons(k),4:5)).^2)),...
        persons);
end
Dt = Dt<=maxDist;
Di = ~diag(true(length(persons),1));
[I1,I2] = meshgrid(persons);
groups = [repmat(X(1,2),[nnz(Di) 1]) I1(Di) I2(Di) Mt(Di) Mt(Di) Dt(Di)];

end

%%
function [ phat, vhat ] = behaviorPredict(method,p,v,u,z,params,groups,dt)
%BEHAVIORPREDICT

if strcmp(method,'lin')
    vhat = v(1,:);
elseif strcmp(method,'ewap')
    vhat = params(6)* v(1,:) + (1-params(6)) *...
            fminunc(@(x) ewapEnergy(x,p,v,u,z,params),...
            v(1,:),...
            optimset('GradObj','off','LargeScale','off','Display','off'));
elseif strcmp(method,'attr')
    vhat = fminunc(@(x) myEnergy(x,p,v,u,z,params,groups),...
           u*(z-v(1,:))/sqrt(sum((z-v(1,:)).^2)),...v(1,:),...
           optimset('GradObj','off','LargeScale','off','Display','off'));
end
% if sqrt(sum(vhat.^2))>2.0 || all(vhat==0), keyboard; end
vhat(isnan(vhat)) = 0;
phat = p(1,:) + vhat * dt;

end

%%
function [ phat, vhat ] = appearanceFeedback(...
    method,pprev,phat,vid,time,H,varargin)
%APPEARANCEFEEDBACK
%
%   [phat,vhat] = appearanceFeedback('mix',p,phat,vid,time,H)
%   [phat,vhat] = appearanceFeedback('mix',p,phat,img1,img2,H)
%
% Input:
%   method: tracking method switch, 'corr' or 'mix'
%    pprev: position at time t
%    phat: initial position at time t+1, if empty pprev is used
%      vid: video object
%     time: 1-by-2 timestamps [tprev tnext]
%        H: homography transformation

% Options
sa = 0.01;   % Variance of appearance prediction
sb = 4.0;   % Variance of error in behavioral prediction (std^2)
dt = 0.4;    % Timestep = 1/fps

tmplSize = [32 16];     % Half of template size
winSize = [40 32];      % Half of ROI size
anchorPos = [-24 0];    % Center position of template patch
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'SigmaA'), sa = varargin{i+1}; end
    if strcmp(varargin{i},'SigmaB'), sb = varargin{i+1}; end
    if strcmp(varargin{i},'TmplSize'), tmplSize = varargin{i+1}; end
    if strcmp(varargin{i},'WinSize'), winSize = varargin{i+1}; end
    if strcmp(varargin{i},'AnchorPos'), anchorPos = varargin{i+1}; end
    if strcmp(varargin{i},'TimeStep'), dt = varargin{i+1}; end
end

% Check input format
if isobject(vid)
    img1 = read(vid,time(1));
    img2 = read(vid,time(2));
else
    img1 = vid;
    img2 = time;
end

% Inverse homography transform (world to pixel)
Pprev = [pprev ones(size(pprev,1),1)] / H';
Pprev = round([Pprev(:,1)./Pprev(:,3) Pprev(:,2)./Pprev(:,3)]);
Pprev(:,1) = Pprev(:,1) + anchorPos(1);
Pprev(:,2) = Pprev(:,2) + anchorPos(2);
Phat = [phat ones(size(phat,1),1)] / H';
Phat = round([Phat(:,1)./Phat(:,3) Phat(:,2)./Phat(:,3)]);
Phat(:,1) = Phat(:,1) + anchorPos(1);
Phat(:,2) = Phat(:,2) + anchorPos(2);

% Range check; if outside the valid pixel region, don't apply
valid = find(...
    Pprev(:,1) - tmplSize(1) >= 1 &...
    Pprev(:,2) - tmplSize(2) >= 1 &...
    Pprev(:,1) + tmplSize(1) <= size(img1,1) &...
    Pprev(:,2) + tmplSize(2) <= size(img1,2) &...
    Phat(:,1) - winSize(1) >= 1 &...
    Phat(:,2) - winSize(2) >= 1 &...
    Phat(:,1) + winSize(1) <= size(img2,1) &...
    Phat(:,2) + winSize(2) <= size(img2,2)...
);

% Estimate motion vector
if ~isempty(valid)
    if strcmp(method,'corr')
        [Phat(valid,:),R] = cvtrack(img1,img2,...
            Pprev(valid,:),Phat(valid,:),tmplSize,winSize);
    elseif strcmp(method,'mix')
        [tmp,R] = cvtrack(img1,img2,...
            Pprev(valid,:),Phat(valid,:),tmplSize,winSize);
        for i = 1:length(valid)
            X = R(R(:,1)==i,2:4);   % Table(px,py,NCC) for tracker i
            lnPa = -((1-X(:,3))/sa).^2;
            lnPb = -(((X(:,1)-Phat(valid(i),1)).^2+...
                      (X(:,2)-Phat(valid(i),2)).^2)/sb);
            lnProb = lnPa + lnPb;
            ind = find(lnProb==max(lnProb),1);
            Phat(valid(i),:) = X(ind,1:2); % argmax of Prob
        end
    elseif strcmp(method,'mixSQ')
        [tmp,R] = cvtrackSQ(img1,img2,...
            Pprev(valid,:),Phat(valid,:),tmplSize,winSize);
        for i = 1:length(valid)
            X = R(R(:,1)==i,2:4);   % Table(px,py,NCC) for tracker i
            lnPa = -X(:,3)/sa;
            lnPb = -(((X(:,1)-Phat(valid(i),1)).^2+...
                      (X(:,2)-Phat(valid(i),2)).^2)/sb);
            lnProb = lnPa + lnPb;
            ind = find(lnProb==max(lnProb),1);
            Phat(valid(i),:) = X(ind,1:2); % argmax of Prob
        end
    else
        error('Unsupported method');
    end
end

% Homography transform (pixel to world)
Phat(:,1) = Phat(:,1) - anchorPos(1);
Phat(:,2) = Phat(:,2) - anchorPos(2);
phat = [Phat ones(size(Phat,1),1)] * H';
phat = [phat(:,1)./phat(:,3) phat(:,2)./phat(:,3)];
vhat = (phat - pprev)/dt;

end
