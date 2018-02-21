function [ err, R ] = simulate( S, method, params, varargin )

%SIMULATE multiple behavioral simulation and error evaluation
%
% Input:
%     S: simulation struct containing
%        dataset: label of the dataset
%            vid: video object
%         frames: S.framesstamps of the simulation
%              H: homography
%           obst: obstacles (px,py)
%           dest: destinations (px,py)
%           trks: tracker instances (id,start,end)
%           obsv: observations (t,id,px,py,vx,vy,dest,u)
%           grps: group relations (t,id,id,label)
%   method: 'none', 'lin', 'ewap', 'attr'
%   params: parameters for the method
% Output:
%   err: error array for each simulation step
%   Pred: result table for each simulation step

% Options
Npast = 5;      % Past steps to use in destination/group prediction
maxDist = 5.0;  % Maximum distance to consider grouping
dt = 0.4;       % Timestep
appMethod = 'mix';      % How to merge appearance and behavioral prediction
appRate = 4;            % Behavior/Appearance sampling rate
predictSpeed = true;    % Either estimate confortable speed or not
thresh = 10;            % Threshold parameter for background subtraction
tmplSize = [32 16];     % Half of template size
anchorPos = [-24 0];    % Center position of template patch
DEBUG = false;
% Destination prediction method
if      strcmp(method,'ewap'), predictDest = 'nn';
elseif  strcmp(method,'attr'), predictDest = 'svm';
elseif  strcmp(method,'attrmc'), predictDest = 'svm';
else                           predictDest = 'none';
end

for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Past'),          Npast = varargin{i+1}; end
    if strcmp(varargin{i},'PredictSpeed'),  predictSpeed = varargin{i+1}; end
    if strcmp(varargin{i},'PredictDest'),   predictDest = varargin{i+1}; end
    if strcmp(varargin{i},'MaxDist'),       maxDist = varargin{i+1}; end
    if strcmp(varargin{i},'AppMethod'),     appMethod = varargin{i+1}; end
    if strcmp(varargin{i},'AppRate'),       appRate = varargin{i+1}; end
    if strcmp(varargin{i},'Threshold'),     thresh = varargin{i+1}; end
    if strcmp(varargin{i},'TmplSize'),      tmplSize = varargin{i+1}; end
    if strcmp(varargin{i},'AnchorPos'),     anchorPos = varargin{i+1}; end
    if strcmp(varargin{i},'DEBUG'),         DEBUG = varargin{i+1}; end
    if strcmp(varargin{i},'class'),         class = varargin{i+1}; end
    
end

%% Initialize variables for simulation

% Initial obsv and grps table
R.obsv = cell(size(S.trks,1),1);
R.grps = cell(size(S.trks,1),1);
for i = 1:size(S.trks,1)
    R.obsv{i} = S.obsv(S.obsv(:,1)>=S.trks(i,2) &...  % from start S.frames
                       S.obsv(:,1)<=S.trks(i,3) &...  % to the end
                       S.obsv(:,2)==S.trks(i,1),:);
    R.grps{i} = S.grps(S.grps(:,1)>=S.trks(i,2) &...  % from start S.frames
                       S.grps(:,1)<=S.trks(i,3) &...  % to the end
                       S.grps(:,2)==S.trks(i,1),:);
end
R.obsv = cell2mat(R.obsv);
R.grps = cell2mat(R.grps);

% Initialize background image
if ~isempty(S.vid) && ~isempty(S.H)
    img = read(S.vid,S.frames(1));
    bg = single(img);
end


%% Start simulation
for j = 1:length(S.frames)-1
    
    % Active trackers at S.frames(j)
    persons = S.trks(S.trks(:,2)<=S.frames(j)&...
                     S.frames(j)<=S.trks(:,3),1)';
    
    %% Predict comfortable speed = running average
    if predictSpeed
        for pid = persons
            R.obsv(R.obsv(:,1)==S.frames(j+1) & R.obsv(:,2)==pid,8) =...
                 mean(sqrt(sum(R.obsv(R.obsv(:,1)>=S.frames(max(1,j-Npast))&...
                                      R.obsv(:,1)<=S.frames(j)&...
                                      R.obsv(:,2)==pid,5:6).^2,2)));
        end
    end

    %% Predict dest
    % TODO: what about prediction at S.frames(end)?
    if strcmp(predictDest,'svm') && ~isempty(S.Cd) % use SVM
        for pid = persons
            % Query (p,v) of pid in the past to the current at most Npast
            X = [S.obsv(S.obsv(:,1)>=S.frames(max(1,j-Npast)) &...
                        S.obsv(:,1)< S.trks(S.trks(:,1)==pid,2) &...
                        S.obsv(:,2)==pid,3:6);...
                 R.obsv(R.obsv(:,1)>=S.frames(max(1,j-Npast)) &...
                        R.obsv(:,1)<=S.frames(j) &...
                        R.obsv(:,2)==pid,3:6)];
            % Compute features
            F = destFeature(X(:,1),X(:,2),X(:,3),X(:,4),...
                            'XBins',S.Cd.XBins,...
                            'YBins',S.Cd.YBins,...
                            'RhoBins',S.Cd.RBins,...
                            'ThetaBins',S.Cd.TBins);
            % Predict labels
            R.obsv(R.obsv(:,1)==S.frames(j)&...
                   R.obsv(:,2)==pid,7) = svmpredict(1,F,S.Cd.C);
        end
    elseif strcmp(predictDest,'nn') % Nearest neighbor prediction
        for pid = persons
            % Query (p,v) of pid at the current time
            X = R.obsv(R.obsv(:,1)==S.frames(j) &...
                       R.obsv(:,2)==pid,3:6);
            % Predict labels
            R.obsv(R.obsv(:,1)==S.frames(j)&...
                   R.obsv(:,2)==pid,7) = nnpredict(X,S.dest);
        end
    end

    %% Predict group
    % TODO: what about prediction at S.frames(end)?
    % TODO: inactive trackers
    if ~isempty(S.Cg)
        pairs = R.grps(R.grps(:,1)==S.frames(j),2:3);
        for k = 1:size(pairs,1)
            % Query (p,v) of person in pairs(k,1)
            X1 = [S.obsv(S.obsv(:,1)>=S.frames(max(1,j-Npast)) &...
                         S.obsv(:,1)< S.trks(S.trks(:,1)==pairs(k,1),2) &...
                         S.obsv(:,2)==pairs(k,1),[1 3:6]);...
                  R.obsv(R.obsv(:,1)>=S.frames(max(1,j-Npast)) &...
                         R.obsv(:,1)<=S.frames(j) &...
                         R.obsv(:,2)==pairs(k,1),[1 3:6])];
            % Query (p,v) of person in pairs(k,2)
            if any(persons==pairs(k,2))
                X2 = [S.obsv(S.obsv(:,1)>=S.frames(max(1,j-Npast)) &...
                             S.obsv(:,1)< S.trks(S.trks(:,1)==pairs(k,2),2) &...
                             S.obsv(:,2)==pairs(k,2),[1 3:6]);...
                      R.obsv(R.obsv(:,1)>=S.frames(max(1,j-Npast)) &...
                             R.obsv(:,1)<=S.frames(j) &...
                             R.obsv(:,2)==pairs(k,2),[1 3:6])];
            else % this person is not included in the tracker list
                X2 =  S.obsv(S.obsv(:,1)>=S.frames(max(1,j-Npast)) &...
                             S.obsv(:,1)<= S.frames(j) &...
                             S.obsv(:,2)==pairs(k,2),[1 3:6]);
            end
            % Predict label if not too distant
            if sqrt(sum((X1(end,3:4) - X2(end,3:4)).^2,2)) <= maxDist
                R.grps(R.grps(:,1)==S.frames(j) &...
                       R.grps(:,2)==pairs(k,1) &...
                       R.grps(:,3)==pairs(k,2),4) =...
                    logical(svmpredict(1,grpFeature(X1,X2),S.Cg.C));
            else
                R.grps(R.grps(:,1)==S.frames(j) &...
                       R.grps(:,2)==pairs(k,1) &...
                       R.grps(:,3)==pairs(k,2),4) = 0;
            end
        end
    end

    %% Predict velocity and position by behavior
    % Get behavioral cue
    curr = R.obsv(:,1)==S.frames(j);           % index
    next = R.obsv(:,1)==S.frames(j+1)  ;         % index
    % (person,phat,vhat,p): persons necessary to be predicted
    subjects = persons(...
        arrayfun(@(pid) any(next & R.obsv(:,2)==pid),persons));
    Pb = [subjects(:) zeros(length(subjects),6)];
    for pid = Pb(:,1)'
        subject = R.obsv(:,2)==pid; % index
        others  = R.obsv(:,2)~=pid; % index
        % Set up group indicator
        groups = logical(arrayfun(@(qid)...
            R.grps(R.grps(:,1)==S.frames(j) &...
                   R.grps(:,2)==pid &...
                   R.grps(:,3)==qid,4),...
            R.obsv(curr & others, 2)));
        % Predict by behavior
        [phat,vhat] = behaviorPredict(method,...
            [R.obsv(curr & subject,3:4); S.obst;...
             R.obsv(curr & others, 3:4)],... % position
            [R.obsv(curr & subject,5:6);zeros(size(S.obst));...
             R.obsv(curr & others, 5:6)],... % velocity
            R.obsv(curr & subject,8),...    % ui
            S.dest(R.obsv(curr & subject,7),:),...    % dest
            params,...                                % params
            [true;false(size(S.obst,1),1);groups],... % groups
            dt);
        % Position
        Pb(Pb(:,1)==pid,2:7) = [phat vhat R.obsv(curr & subject,3:4)];
    end

    %% Mix prediction with the appearance model
    if ~isempty(S.vid) && ~isempty(S.H) && ~strcmp(appMethod,'none')
        img0 = img; % Keep previous frame
        %[img,bg] = cvbgsubtract2(read(S.vid,S.frames(j+1)),bg,j+1,thresh);
         img = read(S.vid,S.frames(j+1)); % If not applying bgsubtract
%         if ~isempty(Pb) &&...
%            ~(strcmp(appMethod,'less') && mod(j,appRate)~=0)
%             [phat,vhat] = appearanceFeedback(appMethod,...
%                 Pb(:,6:7),Pb(:,2:3),img0,img,S.H,'TimeStep',dt,...
%                 'InitialPos',strcmp(method,'none'),varargin{:});
%             Pb(:,2:5) = [phat vhat];
%         end

        % DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if DEBUG && ~isempty(Pb) && ~(strcmp(appMethod,'less') && mod(j,appRate)~=0)
            fprintf('Frame: %d\n',S.frames(j));
            disp(Pb);
            imshow(img);
            pcurr = [Pb(:,6:7) ones(size(Pb,1),1)] / S.H';
            pcurr = [pcurr(:,1)./pcurr(:,3) pcurr(:,2)./pcurr(:,3)];
            phat = [Pb(:,2:3) ones(size(Pb,1),1)] / S.H';
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
                for l = R.grps(R.grps(:,1)==S.frames(j) &...
                               R.grps(:,2)==k & ...
                               R.grps(:,4)==1,3)'
                    plot([pcurr(Pb(:,1)==k,2) pcurr(Pb(:,1)==l,2)],...
                         [pcurr(Pb(:,1)==k,1) pcurr(Pb(:,1)==l,1)],'g--');
                end
            end
            text(5,10,sprintf('dataset=%s method=%s frame=%d',...
                              S.dataset,method,S.frames(j)),'Color','w');
            hold off;
            drawnow; %pause(.5);
        end
        % DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end

    % Update position and velocity
    for pid = Pb(:,1)'
        R.obsv(R.obsv(:,1)==S.frames(j+1) & R.obsv(:,2)==pid,3:6) =...
            Pb(Pb(:,1)==pid,2:5);
    end
end
    
%% Compute distance between predicted and true positions
err = cell(size(S.trks,1),1);
for i = 1:size(S.trks,1)
    ind = find(R.obsv(:,2) == S.trks(i,1) & R.obsv(:,1) > S.trks(i,2));
    err{i} = arrayfun(@(k) sqrt(sum((...
        S.obsv(S.obsv(:,1)==R.obsv(k,1) & S.obsv(:,2)==R.obsv(k,2),3:4)-...
        R.obsv(k,3:4)).^2,2)),...
        ind);
end
err = cell2mat(err);

end


%% SUBROUTINES

%% Behavioral prediction
function [ phat, vhat ] = behaviorPredict(method,p,v,u,z,params,groups,dt)
%BEHAVIORPREDICT

if strcmp(method,'ewap')
    vhat = params(6)* v(1,:) + (1-params(6)) *...
            fminunc(@(x) ewapEnergy(x,p,v,u,z,params),...
            v(1,:),...
            optimset('GradObj','off','LargeScale','off','Display','off'));
elseif strcmp(method,'attr')
    vhat = fminunc(@(x) myEnergy(x,p,v,u,z,params,groups),...
           ...u*(z-v(1,:))/sqrt(sum((z-v(1,:)).^2)),...
           v(1,:),...
           optimset('GradObj','off','LargeScale','off','Display','off'));
elseif strcmp(method,'attrmc')
    vhat = fminunc(@(x) myEnergyMultiClass(x,p,v,u,z,params,groups,class),...
        ...u*(z-v(1,:))/sqrt(sum((z-v(1,:)).^2)),...
        v(1,:),...
        optimset('GradObj','off','LargeScale','off','Display','off'));
else
    vhat = v(1,:);
end
vhat(isnan(vhat)) = 0;
phat = p(1,:) + vhat * dt;

end

%%
function [ phat, vhat ] = appearanceFeedback(...
    method,pprev,phat,vid,time,H,varargin)
%APPEARANCEFEEDBACK
%
%   [phat,vhat] = appearanceFeedback('mix',p,phat,vid,S.frames,H)
%   [phat,vhat] = appearanceFeedback('mix',p,phat,img1,img2,H)
%
% Input:
%   method: tracking method switch, 'corr' or 'mix'
%    pprev: position at S.frames t
%    phat: initial position at S.frames t+1, if empty pprev is used
%      vid: video object
%     S.frames: 1-by-2 S.framesstamps [tprev tnext]
%        H: homography transformation

% Options
s = 1.0e4;   % ratio of variance between app and behavior (s_b^2/s_a^2)
dt = 0.4;    % Timestep = 1/fps

tmplSize = [32 16];     % Half of template size
winSize = [48 36];      % Half of ROI size
anchorPos = [-24 0];    % Center position of template patch
initPos = false;        % Flag to switch ROI from initial Phat
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Sigma'), s = varargin{i+1}; end
    if strcmp(varargin{i},'TmplSize'), tmplSize = varargin{i+1}; end
    if strcmp(varargin{i},'WinSize'), winSize = varargin{i+1}; end
    if strcmp(varargin{i},'AnchorPos'), anchorPos = varargin{i+1}; end
    if strcmp(varargin{i},'TimeStep'), dt = varargin{i+1}; end
    if strcmp(varargin{i},'InitPos'), initPos = varargin{i+1}; end
end

% Check input format
if isobject(vid)
    img1 = read(vid,time(1));
    img2 = read(vid,time(2));
else
    img1 = vid;
    img2 = time;
end

% Set variance of the initial position to be inf
if strcmp(method,'corr'), s = inf; end

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
if initPos
    Phat(valid,:) = Pprev(valid,:);
end

% Estimate motion vector
if ~isempty(valid)
    % Compute correlation
    if strcmp(method,'mixSQ')
        [~,R] = cvtrackSQ(img1,img2,...
            Pprev(valid,:),Phat(valid,:),tmplSize,winSize);
    else
        [~,R] = cvtrack(img1,img2,...
            Pprev(valid,:),Phat(valid,:),tmplSize,winSize);
    end
    % For each valid tracker, compute MLE
    for i = 1:length(valid)
        % Compute log-probability
        X = R(R(:,1)==i,2:4);   % Table(px,py,NCC) for tracker i
        if s == 0
            lnProb = -(((X(:,1)-Phat(valid(i),1)).^2+...
                        (X(:,2)-Phat(valid(i),2)).^2));
        else
            lnPa = -((1-X(:,3))).^2;
            lnPb = -(((X(:,1)-Phat(valid(i),1)).^2+...
                      (X(:,2)-Phat(valid(i),2)).^2)/s);
            lnProb = lnPa + lnPb;
        end
        % Flat peak handling: find the location closest to the center
        ind = find(lnProb==max(lnProb));
        if length(ind)>1
            d = sum(X(ind,1:2) - ones(length(ind),1)*Phat(valid(i),:)).^2;
            ind = ind(find(d==min(d),1));
        end
        % Update
        Phat(valid(i),:) = X(ind,1:2); % argmax of Prob
    end
end

% Homography transform (pixel to world)
Phat(:,1) = Phat(:,1) - anchorPos(1);
Phat(:,2) = Phat(:,2) - anchorPos(2);
phat = [Phat ones(size(Phat,1),1)] * H';
phat = [phat(:,1)./phat(:,3) phat(:,2)./phat(:,3)];
vhat = (phat - pprev)/dt;

end

%%
function [ Y ] = nnpredict( X, dest )
%NNPREDICT nearest neighbor prediction
% Input:
%   X: (px py vx vy)
%   dest: (px py)
% Output:
%   Y: index of dest

phi = atan2(dest(:,2)-X(2),dest(:,1)-X(1));
phi0 = atan2(X(4),X(3));
d = [cos(phi) sin(phi)] * [cos(phi0);sin(phi0)];
Y = find(d==max(d),1);

end
