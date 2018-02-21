function [ D ] = pathPredict( T, Obj, params, n, method )
%PATHPREDICT n-step prediction for ewap model
% T(id,px,py,vx,vy,pdestx,pdesty,u,gid)
% Obj(id,px,py)

if isscalar(n), n = n*ones(1,size(T,1)); end

if strcmp(method,'ewap')
    D = ewapPredict(T,Obj,params,n);
elseif strcmp(method,'attraction')
    D = myPredict(T,Obj,params,n);
else
    D = linPredict(T,n);
end

end

function [D] = ewapPredict( T, Obj, params, n )
%EWAPPREDICT n-step prediction for ewap model
persons = T(:,1)';
D = cell(max(n)+1,1);
D{1} = T(:,1:5);
for i = 1:max(n)
    for id = persons(i<=n)
        Others = D{i}(D{i}(:,1)~=id,:);
        try
            vhat = fminunc(@(x) ewapEnergy(x,...
                                [D{i}(D{i}(:,1)==id,2:3);...
                                 Others(:,2:3);Obj],... % p
                                [D{i}(D{i}(:,1)==id,4:5);...
                                 Others(:,4:5);zeros(size(Obj))],... % v
                                T(T(:,1)==id,8),...     % ui
                                T(T(:,1)==id,6:7),...   % zi
                                params...               % params
                                ),...
                            D{i}(D{i}(:,1)==id,4:5),...  % init value
                            optimset(...
                                'GradObj','off',...
                                'LargeScale','off',...
                                'Display','off'...
                                )...
                        );
        catch ME
            disp(ME.message);
            vhat = [0 0];   % outlier might have strange value
        end
        v = params(6)*D{i}(D{i}(:,1)==id,4:5)+(1-params(6))*vhat;
        p = D{i}(D{i}(:,1)==id,2:3) + v;
        D{i+1} = [D{i+1};id p v];
    end
end
D = cat(1,D{:});

end

function [D] = myPredict( T, Obj, params, n )
%MYPREDICT n-step prediction for attraction model
% T(id,px,py,vx,vy,pdestx,pdesty,u,gid)
persons = T(:,1)';
D = cell(max(n)+1,1);
D{1} = T(:,1:5);
for i = 1:max(n)
    for id = persons(i<=n)
        Others = D{i}(D{i}(:,1)~=id,:);
        groups = arrayfun(@(x) T(T(:,1)==id,9)==T(T(:,1)==x,9),Others(:,1));
        try
            v = fminunc(@(x) myEnergy(x,...
                                [D{i}(D{i}(:,1)==id,2:3);...
                                 Others(:,2:3);Obj],... % p
                                [D{i}(D{i}(:,1)==id,4:5);...
                                 Others(:,4:5);zeros(size(Obj))],... % v
                                T(T(:,1)==id,8),...     % ui
                                T(T(:,1)==id,6:7),...   % zi
                                params,...              % params
                                logical([true; groups; false(size(Obj,1),1)])),...
                        D{i}(D{i}(:,1)==id,4:5),...  % init value
                        optimset(...
                            'GradObj','off',...
                            'LargeScale','off',...
                            'Display','off'...
                            )...
                    );
        catch ME
            disp(ME.message);
            v = [0 0];   % outlier might have strange value
        end
        p = D{i}(D{i}(:,1)==id,2:3) + 0.4 * v;
        D{i+1} = [D{i+1};id p v];
    end
end
D = cat(1,D{:});

end

function [D] = linPredict( T, n )
%LINPREDICT n-step prediction for linear model

% Simple linear extrapolation
persons = T(:,1)';
D = cell(max(n)+1,1);
D{1} = T(:,1:5);
for i = 1:max(n)
    for id = persons(i<=n)
        v = D{i}(D{i}(:,1)==id,4:5);
        p = D{i}(D{i}(:,1)==id,2:3) + 0.4 * v;
        D{i+1} = [D{i+1};id p v];
    end
end
D = cat(1,D{:});

end

