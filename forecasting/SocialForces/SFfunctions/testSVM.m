function [] = testSVM( )
%TESTSVM evaluates SVM performance on paired trajectories classification
%   Path set for mac

addpath('../libsvm-mat-3.0-1/');
load ewap_dataset;
T = seq2table(seq);
[X,Y] = tab2feature(T);

% Training by cross validation
nfolds = 5;
C = cell(nfolds,1);
A = cell(nfolds,1);
for i = 1:nfolds
    ind = mod((1:size(X,1))',nfolds)==i-1;
    % Training
    C{i} = svmtrain(Y(~ind),X(~ind,:));
    % Test
    [Yhat,A{i}] = svmpredict(Y(ind),X(ind,:),C{i});
    fprintf('Precision= %f (/)\n',100*nnz((Yhat>0)&(Y(ind)>0))/nnz(Yhat>0));
    fprintf('Recall   = %f (/)\n',100*nnz((Yhat>0)&(Y(ind)>0))/nnz(Y(ind)>0));
end
A = cellfun(@(x) x(1),A);
Cbest = C{A==max(A)};

% Videos
datasets = fieldnames(seq);
if isfield(seq.(datasets{1}),'video')
    for did = 1:length(datasets)
        aviobj = avifile(['svm_' datasets{did} '.avi'],...
                          'fps',2.5,...
                          'videoname',datasets{did});
        time = unique(T(T(:,1)==did,2))';
        for t = time
            % Retrieve trajectories and calculate features
            persons = T(T(:,1)==did&T(:,2)==t,3)';
            [X,Y,I] = tab2feature(T(T(:,1)==did&...
                                    T(:,2)<=t&...
                                    arrayfun(@(x) any(x==persons),T(:,3)),...
                                    :));
            % Predict
            if ~isempty(X)
                Yhat = svmpredict(Y,X,Cbest);
                groups = I(Yhat>0,2:3);
            else
                groups = ones(0,2);
            end
            mySeqShow(seq.(datasets{did}),groups,t);
            drawnow; aviobj = addframe(aviobj,gca);
        end
        aviobj = close(aviobj);
        close all;
        system(['/opt/local/bin/ffmpeg -i '...
                'svm_' datasets{did} '.avi '...
                '-b 512k -vcodec libx264 -vpre medium '...
                'svm_' datasets{did} '.mp4']);
    end
end

end

%%
function [ T ] = seq2table( seq )
%SEQ2TABLE converts seq structure to table format
%
% Input:
%   seq structure
% Output:
%   T(datasetId,time,person,group,px,py,vx,vy,complete)

datasets = fieldnames(seq);
T = cell(length(datasets),1);

% for each dataset
for did = 1:length(datasets)
    T{did} = iseq2table(seq.(datasets{did}).obsmat,...
                        seq.(datasets{did}).groups,...
                        did);
end

% Merge different datasets
T = cat(1,T{:});

end

function [ T ] = iseq2table(obsmat,groups,did)
%ISEQ2FEATURE creates table format for individual seq structure
%
% Input:
%   obsmat(time,person,px,pz,py,vx,vz,vy)
%   groups(person,group)
%   did
% Output:
%   T(dataset,time,person,group,px,py,vx,vy,complete)

% Rewrite velocity in the original data
T = [ones(size(obsmat,1),1)*did... % append dataset id
          obsmat(:,[1 2])...            % time, person
          arrayfun(@(x) groups(groups(:,1)==x,2),obsmat(:,2))... % group
          obsmat(:,[3 5 6 8])...        % data except z axis
          ones(size(obsmat,1),2)];      % complete observation flag

first = false(size(T,1),1);  % index of first position
persons = unique(T(:,3))';
for i = persons
    % Select tuples of person i (time,id,p_x,p_y,v_x,v_y)
    ind = find(T(:,3)==i);
    t = T(ind,:);
    first(ind(1)) = true; % Remember location
    % Update rows
    v = 2.5*(t(2:end,4:5)-t(1:end-1,4:5));
    if length(ind)>1
        T(ind,7:8) = [v(1,:); v];
    end
end
T(first,end) = false;

end

%%
function [X,Y,I] = tab2feature( T )
% Function to compute features for all possible pairs of tuples
%
% Input:
%   T(datasetId,time,person,group,px,py,vx,vy,complete)
% Output:
%   X(feature1,...,featureN)
%   Y(label)
%   I(dataset,person,person)

datasets = unique(T(:,1))';
X = cell(length(datasets),1);
Y = cell(length(datasets),1);
I = cell(length(datasets),1);

% Find potential pairs (domain) and grand truth (label)
for did = datasets
    % Query person id of dataset did
    persons = unique(T(T(:,1)==did,3))';
    M = false(length(persons));
    G = false(length(persons));
    % for each person pid
    for pid = persons
        % retrieve tuples of person pid
        Ti = T(T(:,1)==did & T(:,3)==pid & T(:,9)==true,:);
        % query any tuples sharing time with pid but not of himself
        To = T(T(:,1)==did & T(:,3)~=pid & T(:,9)==true &...
               arrayfun(@(x) any(x==Ti(:,2)),T(:,2)),:);
        % get a list of other persons possibly related to person pid
        others = unique(To(:,3))';
        % mark the possible pairs
        for oid = others
            M(persons==pid,persons==oid) = true;
        end
        % mark the true pairs
        groups = unique(To(To(:,4)==Ti(1,4),3))';
        for oid = groups
            G(persons==pid,persons==oid) = true;
        end
    end
    
    % Compute features over possible pairs
    Y{did} = G(M)-(~G(M));          % true labels
    X{did} = zeros(nnz(M),myFeature('len'));
    [I1,I2] = find(M);
    I{did} = [did*ones(size(I1,1),1) persons(I1)' persons(I2)'];
    for i = 1:size(I{did},1)
        X{did}(i,:) = myFeature(T(T(:,1)==did &...
                                  T(:,3)==I{did}(i,2) &...
                                  T(:,9)==true,[2 5:8]),...
                                T(T(:,1)==did &...
                                  T(:,3)==I{did}(i,3) &...
                                  T(:,9)==true,[2 5:8]));
    end
end

% Reshape
X = cat(1,X{:});
Y = cat(1,Y{:});
I = cat(1,I{:});

end

%%
function [X] = myFeature(Ti,Tj)
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

function [ ] = mySeqShow( seq, groups, frame )
%MYSEQSHOW draw a scene at a specified frame with group label
%    seq: seq struct
% groups: edges (pid,pid)
%  frame: frame id

Ncolor = 16;
C = lines(Ncolor);

% Table(time,id,px,pz,py,vx,vz,vy)
D = seq.obsmat(seq.obsmat(:,1)==frame,:);

% Compute positions in image coordinates
P = [D(:,3) D(:,5) ones(size(D,1),1)] / seq.H';
P = [P(:,1)./P(:,3) P(:,2)./P(:,3)];

% Set group id
G = conn(D(:,2),groups);
cind = arrayfun(@(x)find(cellfun(@(y)any(x==y),G)),D(:,2));
edges = [(1:size(D,1))' (1:size(D,1))';...
    arrayfun(@(x) find(D(:,2)==x),groups(:,1))...
    arrayfun(@(x) find(D(:,2)==x),groups(:,2))];

% Plot
imshow(read(seq.video,frame));
hold on;
plot(seq.static(:,1),seq.static(:,2),'w+');
for i = 1:size(edges,1)
    plot(P(edges(i,1:2),2),P(edges(i,1:2),1),'-o',...
         'Color',C(cind(edges(i,1)),:));
end
for i = 1:size(D,1)
    text(P(i,2),P(i,1)+10,cellstr(num2str(D(i,2))),'Color',C(cind(i),:));
end
text(5,10,sprintf('Frame %d',frame),'Color','w');
hold off;

end


function [ C ] = conn( V, E )
%CONN return connected components of the graph
% V: N-by-1 rows of vertices with identifier for each
% E: M-by-2 rows of edges

mark = true(size(V)); % indicator for unvisited nodes
C = {};
k = 1;
while any(mark)
    first = find(mark,1);
    mark(first) = false;
    C{k} = unique([V(first)...
                   E(E(:,1)==V(first),2)'...
                   E(E(:,2)==V(first),1)']);
    ind = arrayfun(@(x)any(x==C{k}),V);
    while any(mark(ind))
        mark(ind) = false;
        C{k} = unique([C{k}...
                       E(arrayfun(@(x)any(x==C{k}),E(:,1)),2)'...
                       E(arrayfun(@(x)any(x==C{k}),E(:,2)),1)']);
        ind = arrayfun(@(x)any(x==C{k}),V);
    end
    k = k + 1;
end
end
