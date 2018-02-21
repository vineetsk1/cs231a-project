%% main script for multiple tracking

% Load and convert dataset
D = importData();
%D = D.D;

% Create tables:
%   Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Obst(dataset,px,py)
%   Dest(dataset,px,py)
[Obsv, Obst, Dest] = data2table(D);
%save;

%load;
addpath('libsvm-mat-3.0-1/');
addpath('cv');

%% Train SVM classifier
TrainSet = 3;
Ninterval = 12;
Npast = 6;

fprintf('\n=== Training SVMs ===\n');
fprintf('Group classifier\n');
% Group classifier
[Obs, Sims] = obsv2sim(Obsv(Obsv(:,1)==TrainSet,:),...
    'Interval',Ninterval,...    % Sampling rate
    'Duration',0,...            % Future not needed
    'Past',Npast);              % Past steps to use
Cg = grpTrain2(Obs,Sims);
fprintf('Destination classifier\n');
% Destination classifier
Cd = destTrain(Obs,Sims);
if isempty(Cd{3})&&~isempty(Cd{4}), Cd{3} = Cd{4}; end

%% Global config
TestSet = 3;

params = {...
    [1.034086 1.068818 0.822700 0.940671 1.005634 0.842588],...
    [0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420]...
};
fprintf('  Params:\n');
fprintf('    EWAP:');
for i = 1:length(params{1}), fprintf(' %f',params{1}(i)); end
fprintf('\n');
fprintf('    ATTR:');
for i = 1:length(params{2}), fprintf(' %f',params{2}(i)); end
fprintf('\n');

% Create simulation tables:
%   Obs(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Sims(simulation,dataset,person,start,duration)

% Arrange the structure
Obsv = Obsv(Obsv(:,1)==TestSet,2:end);
Obs = [ones(size(Obsv,1),1) Obsv];

% Set uniform speed preference
Obs(:,9) = median(Obs(:,9));

% Crop out-of-bound observations
winSize = [36 36];
tmplSize = [24 16];%[24 12];
anchorPos = [-28 0];
% anchorPos = [-24 0];
P = Obs(:,4:5);
P = [P ones(size(P,1),1)] / D(TestSet).H';
P = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
Obs(P(:,1)<2*winSize(1) | P(:,1)>D(TestSet).video.Height-2*winSize(1) |...
    P(:,2)<2*winSize(2) | P(:,2)>D(TestSet).video.Width -2*winSize(2),:) = [];

% Create sims
time = unique(Obs(:,2));
persons = unique(Obs(:,3));
start = arrayfun(@(pid) Obs(find(Obs(:,3)==pid,1,'first'),2),persons);
last = arrayfun(@(pid) Obs(find(Obs(:,3)==pid,1,'last'),2),persons);
duration = arrayfun(@(i) find(time==last(i))-find(time==start(i)),...
                    (1:length(start))');
Sims = [ones(length(persons),1) TestSet*ones(length(persons),1)...
        persons start duration];

% % Set the start velocity to zero
% for i = 1:length(persons)
%     Obs(Obs(:,2)==start(i) & Obs(:,3)==persons(i),6:7) = [0 0];
% end
clear time persons start last duration;

%% Evaluate error for each methods
methods = {'LIN      ','EWAP     '};
%'EWAP+DST ','ATTR     ','ATTR+DST ','ATTR+GRP ','ATTR+D&G '};
Err = cell(1,length(methods));
Res = cell(1,length(methods));
Grp = cell(1,length(methods));

[Err{1}, Res{1}, Grp{1}] = simErrorM(Obs,Sims,'attr',...
    'Params',params{2},'Obst',Obst,'Dest',Dest,...
    'DestClassifier',Cd,'GroupClassifier',Cg,...
    'Video',D(Sims(1,2)).video,'H',D(Sims(1,2)).H,'PredictSpeed',true,...
    'WinSize',winSize,'TmplSize',tmplSize,'AnchorPos',anchorPos,...
...    'SigmaA',0.07,'SigmaB',500,'AppMethod','mix');
    'SigmaA',1,'SigmaB',3200,'AppMethod','mixSQ');


% Plot
C = lines(length(Res));
clf; hold on;
for i = 1%:length(Res)
    plot(Res{i}(:,5),Res{i}(:,4),'+','Color',C(i,:));
end
hold off;
