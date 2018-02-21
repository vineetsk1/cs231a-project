%% Main9: multiple simulation and error evaluation

addpath('libsvm-mat-3.0-1/');
addpath('cv');

%% Prepare simulation data
if true
%     load dataset;
    [Dtest,Dtrain] = datasplit(importData);
%     [Dtrain,Dtest] = datasplit(importData);
    
    % Training configuration of SVMs
    fprintf('\n=== Training SVMs ===\n');
    TrainSet = [1 2 3 4 5]; % Dataset used for group SVMs
    Ninterval = 4;
    Npast = 5;
    
    % Prepare training data
    [Obs, Sims] = obsv2sim(data2table(Dtrain),...
        'Interval',Ninterval,...    % Sampling rate
        'Duration',0,...            % Future not needed here
        'Past',Npast);              % Past steps to use

    % Group classifier
    fprintf('Group classifier\n');
    fprintf('  Training set:');
    fprintf(' %d', TrainSet); fprintf('\n');
    fprintf('  Ninterval: %d\n', Ninterval);

    fprintf('  Npast: %d\n', Npast);
    Cg = grpTrain2(Obs,Sims(arrayfun(@(x) any(x==TrainSet),Sims(:,2)),:));

    % Destination classifiers are prepared for all scenes
    fprintf('Destination classifier\n');
    fprintf('  Ninterval: %d\n  Npast: %d\n', Ninterval, Npast);
    Cd = destTrain(Obs,Sims);

    % Prepare simulation structure
    fprintf('\n=== Subsampling datasets ===\n');
    Datasets = [1 2 3 4 5];
    Ninterval = 16;
    Nduration = 24;
    Ooffset = 5;   % 5 frames to accumulate past before start
    S = data2sims(Dtest,...
        'Datasets',Datasets,...
        'GroupClassifier',Cg,...
        'DestClassifier',Cd,...
        'Ninterval',Ninterval,...
        'Nduration',Nduration,...
        'Ooffset',Ooffset);
    save simulation.mat S;
end

%% Do simulation
if true
    clear;
    DEBUG = false;
    load simulation;    % Get struct array S

    % Simulation configurations
    fprintf('=== MULTIPLE SIMULATION ===\n');
    config = {...
        %'none', 'corr', [];...
        %'lin', 'mix', [];...
        'ewap', 'mix',[1.034086 1.068818 0.822700 0.940671 1.005634 0.842588];...
        'attr', 'mix',[0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420];...
        'lin', 'less', [];...
        'ewap', 'less',[1.034086 1.068818 0.822700 0.940671 1.005634 0.842588];...
        'attr', 'less',[0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420];...
...        'lin', 'none', [];...
...        'ewap', 'none',[1.034086 1.068818 0.822700 0.940671 1.005634 0.842588];...
...        'attr', 'none',[0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420];...    
    };

    err = cell(1,size(config,1));
    R = cell(length(S),size(config,1));
    for j = 1:size(config,1)
        fprintf('Method: %s+%s\n',config{j,1},config{j,2});
        fprintf('  Params: '); fprintf(' %f',config{j,3}); fprintf('\n  ');
        e = cell(length(S),1);
        r = cell(length(S),1);
        tic;
        if DEBUG
            for i = 1:length(S)
                [e{i},r{i}] = simulate(S(i),config{j,1},config{j,3},...
                    'AppMethod',config{j,2},'DEBUG',true);
            end
        else
            parfor i = 1:length(S)
                [e{i},r{i}] = simulate(S(i),config{j,1},config{j,3},...
                    'AppMethod',config{j,2});
            end
        end
        toc;
        err{j} = cell2mat(e);   % Assuming only 1 dataset is contained
        R(:,j) = r(:);
        fprintf('  Error: %f\n', mean(cell2mat(err(:,j))));
    end
    fprintf('\n');

    save main9resultF4.mat S err R config
end

%% Create a report
clear;
load main9resultF4.mat


maxDist = 0.5;
stepRange = [13 13;25 25;];
fprintf('Maximum Distance: %1.2f (m)\n',maxDist);
% Drop irrelevant dataset to check
for j = 1:size(stepRange,1)
    fprintf('Tracking Step: [%d %d]\n',stepRange(j,1),stepRange(j,2));
    for Dataset = {'zara01','zara02','students03'}
        fprintf('%s\n',Dataset{:});
        ind = arrayfun(@(s) ~strcmp(s.dataset,Dataset),S);
        s = S(ind);
        r = R(ind,:);

        % Analyze
        L = salvage(s,r,config,'MaxDist',maxDist,'StepRange',stepRange(j,:));

        % Show
        SUCCESS = sum(cellfun(@(l) nnz(l(:,3)~=0&l(:,2)==l(:,3)), L));
        LOST_COUNT = sum(cellfun(@(l) nnz(l(:,3)==0), L));
        ID_SWITCH = sum(cellfun(@(l) nnz(l(:,3)~=0&l(:,2)~=l(:,3)), L));
        fprintf('  Successful count: ');
        fprintf(' %5d',SUCCESS); fprintf('\n');
        fprintf('  Lost observation: ');
        fprintf(' %5d',LOST_COUNT); fprintf('\n');
        fprintf('  ID switch counts: ');
        fprintf(' %5d',ID_SWITCH); fprintf('\n');
        fprintf('  Failed total:     ');
        fprintf(' %5d',LOST_COUNT+ID_SWITCH); fprintf('\n');
    end
end
