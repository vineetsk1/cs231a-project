% Set up datasets
clear all;addpath('SFfunctions')
%load dataset.mat;
%[D,T,Obj] = importData();

load D.mat;
Datasets = 2;

[Dtrain,Dtest] = datasplit(D);
Strain = data2sims(Dtrain,'SampleMethod','single',...
    'Datasets',Datasets,'Ninterval',4,'Nduration',12);
Stest = data2sims(Dtest,'SampleMethod','single',...
    'Datasets',Datasets,'Ninterval',4,'Nduration',12);
clear D Dtrain Dtest;

% Initial params
params = {...
[1.034086 1.068818 0.822700 0.940671 1.005634 0.842588],...
[0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420],...
};
lb = {[0 0 0 0 0 0],[0 0 0 0 0 0 0 0]};
ub = {[inf inf inf inf inf 1.0],[inf inf inf inf inf inf inf inf]};

% Main loop
Nround = 5;
Nsubset = 5;
Niter = 10;
Elin = mean(simError2(Stest,'lin',[]));
for j = 1:Nround
    fprintf('== Round %d ==\n',j);
    for i = 1:Nsubset
        ind = mod(1:length(Strain),Nsubset)+1==i;   % Subsample index
        % EWAP
        tic;
%         params{1} = fmincon(@(x) sum(simError2(Strain(ind),'ewap',x)),params{1},...
%             [],[],[],[],lb{1},ub{1},[],...
%             optimset('Algorithm','sqp','Display','iter','GradObj','off','MaxIter',Niter));
        params{1} = lsqnonlin(@(x) simError2(Strain(ind),'ewap',x),params{1},...
            lb{1},ub{1},...
            optimset('Display','iter','MaxIter',Niter));
        toc;
        fprintf('ewap: '); fprintf(' %f',params{1}); fprintf('\n');
        % ATTR
        tic;
        params{2} = fmincon(@(x) sum(simError2(Strain(ind),'attr',x)),params{2},...
            [],[],[],[],lb{2},ub{2},[],...
            optimset('Algorithm','sqp','Display','iter','GradObj','off','MaxIter',Niter));
%         params{2} = lsqnonlin(@(x) simError2(Strain(ind),'attr',x),params,...
%             lb{1},ub{1},...
%             optimset('Display','iter','MaxIter',Niter));
        toc;
        fprintf('attr: '); fprintf(' %f',params{2}); fprintf('\n');
        % Check status
        Eewp = mean(simError2(Stest,'ewap',params{1}));
        Eatr = mean(simError2(Stest,'attr',params{2}));
        fprintf('average error: %f %f %f\n', Elin,Eewp,Eatr);
        fprintf('improvement:   %f %f %f\n\n',...
            (Elin-Elin)/Elin,(Elin-Eewp)/Elin,(Elin-Eatr)/Elin);
    end
end
save params.mat params;