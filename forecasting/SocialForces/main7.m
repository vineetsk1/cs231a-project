%% Main script for single-pedestrian simulation
clear all
load('Cd.mat');
load('Cg.mat');
% Load and convert dataset
D = importData();
%save dataset;
% D = load('dataset.mat','D');
%D = D.D;

% Create tables:
%   Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Obst(dataset,px,py)
%   Dest(dataset,px,py)
[Obsv, Obst, Dest] = data2table(D);

addpath('libsvm-mat-3.0-1/');

% Global config
Npast = 8;



load SVMclass
% %[SVMclass,clust]=MultiClassTraining(D);
%%
for TestSet = 6;
    
    fprintf('=== SVM Config ===\n');
    fprintf('  Ninterval: %d\n', Ninterval);
    fprintf('  Npast: %d\n', Npast);
    
    %% Train SVM classifier
    %
    % fprintf('\n=== Training SVMs ===\n');
    % fprintf('Group classifier\n');
    % % Group classifier
    % [Obs, Sims] = obsv2sim(Obsv,'Interval',Ninterval,...    % Sampling rate
    %                             'Duration',0,...            % Future not needed
    %                             'Past',Npast);              % Past steps to use
    % [Cg,R] = grpTrain2(Obs,Sims);
    % fprintf('Destination classifier\n');
    % % Destination classifier
    % Cd = destTrain(Obs,Sims);
    
    
    
    
    %% Learn parameters
    
    for nduration = 12
        Ninterval = 12;           % Interval to subsample simulation sequence
        Nduration = nduration;    % Simulation duration
        fprintf('=== Simulation Config ===\n');
        fprintf('  Ninterval: %d\n', Ninterval);
        fprintf('  Nduration: %d\n', Nduration);
        
        % let's use precomputed values...
        params = {...
            [1.034086 1.068818 0.822700 0.940671 1.005634 0.842588],...
            [0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420],...
            ...[0.694989 0.654856 0.353771 0.993633 1.866139 0.844283],... %10.824449
            ...[0.800902 0.624982 1.332215 1.035327 0.239304 0.029079 2.248789 0.140373 0.584072]... %17.376034
            };
        fprintf('  Params:\n');
        fprintf('    EWAP:');
        for i = 1:length(params{1}), fprintf(' %f',params{1}(i)); end
        fprintf('\n');
        fprintf('    ATTR:');
        for i = 1:length(params{2}), fprintf(' %f',params{2}(i)); end
        fprintf('\n');
        
        %% Evaluate path prediction error
        
        fprintf('\n=== Evaluating simulation ===\n');
        
        % Create simulation tables:
        %   Obs(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
        %   Sims(simulation,dataset,person,start,duration)
        time = unique(Obsv(Obsv(:,1)==TestSet,2));
        [Obs, Sims] = obsv2sim(Obsv(Obsv(:,1)==TestSet&Obsv(:,2)<=median(time),:),... % Test dataset
            'Interval',Ninterval,...    % Sampling rate
            'Duration',Nduration,...    % Simulation duration
            'Past',Npast);              % Must be the same to C
        % Evaluate error for each methods
        methods = {'LIN      ','LTA     ','LTA+DST ','ATTR     ',...
            'ATTR+DST ','ATTR+Grp{TestSet} ','ATTR+D&G ','ATTRMC+D&G','ATTRMC'};
        Err{TestSet} = cell(1,length(methods));
        Errd{TestSet} = cell(1,length(methods));
        Res{TestSet} = cell(1,length(methods));
        Grp{TestSet} = cell(1,length(methods));
        
        %    DEBUG
        
        
        fprintf('%s',methods{1}); tic;
%        [Err{TestSet}{1},Errd{TestSet}{1}, Res{TestSet}{1}, Grp{TestSet}{1}] = simError(Obs,Sims,'lin');
 %       toc; 
 fprintf('%s',methods{2}); tic;
        [Err{TestSet}{2},Errd{TestSet}{2}, Res{TestSet}{2}, Grp{TestSet}{2}] = simError(Obs,Sims,'ewap',...
            'Params',params{1},'Obst',Obst,'Dest',Dest);
        toc;
        %     fprintf('%s',methods{3}); tic;
        %     [Err{TestSet}{3}, Errd{TestSet}{3},Res{TestSet}{3}, Grp{TestSet}{3}] = simError(Obs,Sims,'ewap',...
        %         'Params',params{1},'Obst',Obst,'Dest',Dest,'DestClassifier',Cd);
        %     toc;
        fprintf('%s',methods{4});     tic;
        [Err{TestSet}{4}, Errd{TestSet}{4},Res{TestSet}{4}, Grp{TestSet}{4}] = simError(Obs,Sims,'attr',...
            'Params',params{2},'Obst',Obst,'Dest',Dest);
        toc;
        %     fprintf('%s',methods{5}); tic;
        %     [Err{TestSet}{5}, Res{TestSet}{5}, Grp{TestSet}{5}] = simError(Obs,Sims,'attr',...
        %         'Params',params{2},'Obst',Obst,'Dest',Dest,'DestClassifier',Cd);
        %     toc; fprintf('%s',methods{6}); tic;
        %     [Err{TestSet}{6}, Res{TestSet}{6}, Grp{TestSet}{6}] = simError(Obs,Sims,'attr',...
        %         'Params',params{2},'Obst',Obst,'Dest',Dest,'GroupClassifier',Cg);
        %     toc; fprintf('%s',methods{7});
        tic;
        [Err{TestSet}{7}, Errd{TestSet}{7},Res{TestSet}{7}, Grp{TestSet}{7}] = simError(Obs,Sims,'attr',...
            'Params',params{2},'Obst',Obst,'Dest',Dest,...
            'DestClassifier',Cd,'GroupClassifier',Cg);
        toc; fprintf('\n');
        [Err{TestSet}{8},Errd{TestSet}{8}, Res{TestSet}{8}, Grp{TestSet}{8}] = simError(Obs,Sims,'attrmc',...
            'Params',params{2},'Obst',Obst,'Dest',Dest,...
            'DestClassifier',Cd,'GroupClassifier',Cg,'BehaviorClassifier',SVMclass{TestSet},'Clust',clust);
        toc; fprintf('\n');
        [Err{TestSet}{9},Errd{TestSet}{9}, Res{TestSet}{9}, Grp{TestSet}{9}] = simError(Obs,Sims,'attrmc',...
            'Params',params{2},'Obst',Obst,'Dest',Dest,...
            'BehaviorClassifier',SVMclass{TestSet},'Clust',clust);
        toc; fprintf('\n');
    end
end
%% Report
for TestSet=1:5
    
    
    fprintf('\n=== Results TestSet %f Nduration % f ===\n',TestSet,Nduration);
    L =[1 2 4 7 8 9];
    % Header
    fprintf('||Method  ');
    for i = L%1:length(methods)
        fprintf('||%s',methods{i});
    end
    fprintf('||\n');
    
    % Average error
    E = zeros(1,length(Err{TestSet}));
    fprintf('||Error(m)');
    for i = L%1:length(Err{TestSet})
        E(i) = mean(Err{TestSet}{i});
        fprintf('||% f',E(i));
    end
    fprintf('||\n');
    
    % Average error
    %     fprintf('||Improve ');
    %     for i = L%1:length(Err{TestSet})
    %         fprintf('||% f',(E(1)-E(i))/E(1));
    %     end
    %     fprintf('||\n');
    
    % Percentage Error(under 0.5m)
    
    fprintf('||Percent 0.5');
    for i = L%1:length(Err{TestSet})
        fprintf('||% f',(sum(Err{TestSet}{i}<0.5)/length(Err{TestSet}{i})));
    end
    fprintf('||\n'); fprintf('||Error Final Position ');
    for i = L%1:length(Err{TestSet})
        Ed(i) = mean(Errd{TestSet}{i});
        fprintf('||% f',Ed(i));
    end
    fprintf('||\n');
    % Destination accuracy
    %     fprintf('||Dest Acc');>
    %     for i = 1:length(Res{TestSet})
    %         E = false(size(Res{TestSet}{i},1),1);
    %         for j = 1:size(Res{TestSet}{i},1)
    %             x = Obs(Obs(:,1)==Res{TestSet}{i}(j,1),:);
    %             E(j) = x(x(:,2)==Res{TestSet}{i}(j,2)&...
    %                      x(:,3)==Res{TestSet}{i}(j,3),8)==Res{TestSet}{i}(j,8);
    %         end
    %         fprintf('||% f',nnz(E)/size(Res{TestSet}{i},1));
    %     end
    %     fprintf('||\n');
end
% Group accuracy
%     fprintf('||Group Ac');
%     for i = 1:length(Grp{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet})
%         fprintf('||% f',nnz(Grp{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{i}(:,4)==Grp{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{i}(:,5))/size(Grp{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{TestSet}{i},1));
%     end
%     fprintf('||\n');


%save main7result.mat