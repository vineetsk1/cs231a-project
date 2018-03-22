warning('off','all');

% T(id,px,py,vx,vy,pdestx,pdesty,u,gid)
% D(id,px,py,vx,vy)
% person id, person x, person y, velocity x, velocity y, destination x, destination y, comfortable speed, group id
T=[[1,100,200,5,6,500,800,1,1]];
    % [2,150,250,5,6,500,800,1,2];
    % [3,110,210,-5,-6,500,800,1,3]];

params = {[1.34086 1.068818 0.852700 0.0840671 .00999 0.442588 0.0],...
            [1.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420]};

params2 = {...
    [],...
    [0.211506 3.285591 3.724895 4.234454 2.189293 0.839590],...
    [0.000000 0.000000 0.125332 0.287676 0.000000 0.059122 1.079154 0.216176]...
};

% Converge for one step

% for j=1:12

% Obj(Obj(:,1)==did,2:3)
n = 12
Pewp = pathPredict(T(:,:),[],params{1},n,'ewap');
display(Pewp)
Pewp = pathPredict(T(:,:),[],params{2},n,'ewap');
display(Pewp)

Pewp = pathPredict(T(:,:),[],params2{2},n,'ewap');
display(Pewp)
Pewp = pathPredict(T(:,:),[],params2{3},n,'ewap');
display(Pewp)


% Pewp = pathPredict(T(:,:),[],params{1},n,'attraction');
% display(Pewp)
% Pewp = pathPredict(T(:,:),[],params{2},n,'attraction');
% display(Pewp)

% Pewp = pathPredict(T(:,:),[],params{1},n,'linear');
% display(Pewp)
% Pewp = pathPredict(T(:,:),[],params{2},n,'linear');
% display(Pewp)


% P1=pathPredict(T(:,:),[],params{1},1,'ewap');
% for i=1:300
%     P2=pathPredict([P1(4:6,:),T(:,6:9)],[],params{1},1,'ewap');
%     P1=pathPredict([P2(4:6,:),T(:,6:9)],[],params{1},1,'ewap');
% end
% T(:,2:3) = P1(4:6,2:3);

% display(T(:, 1:3))

% end


% display(T)
% display(P1)
% display(T(:,2))
% display(P1(4:6,2:3))
% display(T)
% display(P1(:,:))