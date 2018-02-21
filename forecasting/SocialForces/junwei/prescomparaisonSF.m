
% Script to compare different predictions

%% Load and preprocess dataset
if ~exist('ewap_dataset.mat','file')
    seq = ewapLoad('ewap_dataset');
    save ewap_dataset.mat seq;
end
load ewap_dataset.mat;

% sequcy = ucyLoad();


[D,Obj] = seq2ewap(seq);
% D(dataset,t,id,px,py,vx,vy,pnextx,pnexty,pdestx,pdesty,u)
% O(id,px,py,vx,vy,groups)



%% For each data point, give a prediction

%
% Params from the original model
% ||Fold||  sigma_d||  sigma_w||     beta|| lambda_1|| lambda_2||    alpha||
% ||   1|| 0.095096|| 2.947568|| 1.350993|| 2.968880|| 0.042497|| 0.690491||
% ||   2|| 0.211506|| 3.285591|| 3.724895|| 4.234454|| 2.189293|| 0.839590||
% ||   3|| 0.144601|| 4.565162|| 3.334477|| 4.712200|| 0.353668|| 0.726025||
% ||Init|| 0.130065|| 2.088003|| 2.085519|| 2.073509|| 1.461433|| 0.729837||
% Avg. Error
% ||Fold||LTA   (m)||LIN   (m)||Improvement (%)||
% ||   1|| 0.392453|| 0.517587|| 24.176398||
% ||   2|| 0.400791|| 0.516708|| 22.433765||
% ||   3|| 0.388444|| 0.530267|| 26.745613||
% ||Avg.|| 0.393896|| 0.521521|| 24.451925||


% Params from the attraction modelParams
% ||Fold||  sigma_d||  sigma_w||     beta|| lambda_1|| lambda_2|| lambda_3|| lambda_4|| lambda_5||
% ||   1|| 0.000000|| 0.000000|| 0.125332|| 0.287676|| 0.000000|| 0.059122|| 1.079154|| 0.216176||
% ||   2|| 1.196977|| 0.003540|| 0.465781|| 0.213274|| 0.000000|| 0.059464|| 0.811728|| 0.212438||
% ||   3|| 0.621705|| 5.933826|| 1.762825|| 3.830825|| 0.377449|| 0.058362|| 3.758142|| 0.205888||
% ||Init|| 0.130081|| 2.087902|| 2.327072|| 2.073200|| 1.461249|| 0.059400|| 1.127248|| 0.215000||
% Avg Error
% ||Fold||LTA   (m)||LIN   (m)||Improvement (%)||
% ||   1|| 0.370219|| 0.517587|| 28.472088||
% ||   2|| 0.348916|| 0.516708|| 32.473340||
% ||   3|| 0.382789|| 0.530267|| 27.812145||
% ||Avg.|| 0.367308|| 0.521521|| 29.585857||


datasets = fieldnames(seq);
methods = {'KK','TrajClustering','Adv SF'};
params = {...
    [],...
    [0.211506 3.285591 3.724895 4.234454 2.189293 0.839590],...
    [0.000000 0.000000 0.125332 0.287676 0.000000 0.059122 1.079154 0.216176]...
};


% t = 2142, 3102:6:3198
vid = true;
if vid, aviobj = avifile('outputhotel.avi','fps',1,'quality',100); end
did = 2;  % defini quel database regarder (did = 1 ETH ;did = 2  HOTEL )

Times = unique(D(D(:,1)==did,2))';j=0; 

for t = 1:300%D(D(:,1)==did & D(:,3)==50,2)'%Times(1:end)% '%%3102:6:3198%Times(100:5:length(Times))
    % Retrieve observations at time t of dataset did
    
    T = D(D(:,1)==did & D(:,2)==t,:);
    if isempty(T), disp(t); continue; end
    % Persons
    persons = T(:,3)';
    % Set #steps to predict
    tend = (arrayfun(@(x) max(D(D(:,1)==did & D(:,3)==x,2)),persons));
    n = (tend - t)/6 + 1;
    n(n>12)=12;
    % Get ground truth
    Ptru = [];
    for i = 1:length(persons)
        pv = D(D(:,1)==did & D(:,2)>=t & D(:,2)<=t+6*n(i) & D(:,3)==persons(i),4:7);
        Ptru = [Ptru;repmat(persons(i),[size(pv,1) 1]) pv];
    end
    % Get lin
    Plin = pathPredict(T(:,3:7),[],[],n,'lin');
    % Get ewap
    Pewp = pathPredict(T(:,[3:7 10:12]),Obj(Obj(:,1)==did,2:3),params{2},n,'ewap');
    % Get lin
    groups = seq.(datasets{did}).groups;
    Tg = arrayfun(@(id) groups(groups(:,1)==id,2),T(:,3));
    Patr = pathPredict(T(:,[3:7 10:12 14]),Obj(Obj(:,1)==did,2:3),params{3},n,'attraction');
    
    
    % Show
    seqShow(seq.(datasets{did}),groups,t);
    
    
    H = seq.(datasets{did}).H;
    for id = persons
        ptru = [Ptru(Ptru(:,1)==id,2:3) ones(nnz(Ptru(:,1)==id),1)]/H';
        plin = [Plin(Plin(:,1)==id,2:3) ones(nnz(Plin(:,1)==id),1)]/H';
        pewp = [Pewp(Pewp(:,1)==id,2:3) ones(nnz(Pewp(:,1)==id),1)]/H';
        patr = [Patr(Patr(:,1)==id,2:3) ones(nnz(Patr(:,1)==id),1)]/H';
        ptru = ptru(:,1:2)./repmat(ptru(:,3),[1 2]);
        plin = plin(:,1:2)./repmat(plin(:,3),[1 2]);
        pewp = pewp(:,1:2)./repmat(pewp(:,3),[1 2]);
        patr = patr(:,1:2)./repmat(patr(:,3),[1 2]);
        % Plot
        
        hold on;
        text(5,10,sprintf('Frame %d',t),'Color','w');
        h0 = plot(ptru(:,2),ptru(:,1),'-w');
        h1 = plot(plin(:,2),plin(:,1),'-b');
        h2 = plot(pewp(:,2),pewp(:,1),'-g');
        h3 = plot(patr(:,2),patr(:,1),'-r');
%         legend([h0 h1 h2 h3],['TRU' methods]);
        hold off;
    end
    % Write
    
    drawnow;
    if vid, aviobj = addframe(aviobj,gca); end
    
end
close;
if vid
    aviobj = close(aviobj);
    system(['/opt/local/bin/ffmpeg -i outputhotel.avi '...
            '-b 512k -vcodec libx264 -vpre medium output.mp4']);
end
