

clear all
close all

%%
addpath('FastMarching_version3b','SFfunctions')
for w=1:1
    
    fprintf([num2str(w) '\n'])
    clear TrS Data D Ti Pedrem N Nind dloc npedi LocalGoal DepGoal PedStart Path Obst r
    %dest=[0 10;0 30;50 30;50 10];
    cd /Users/Amir/Desktop/Silvio/capri4/eccv_alex/SocialForces
    %dest = [0 25;50 25];
    dest = [25 0;25 50];
    params = {[.534086 2.68818 0.0822700 200.940671 .505634 0.842588],...
        [1.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420]};
    method='ewap';
    try
        % filename = 'C:\Users\Alexandre\Dropbox\Stages\Stanford\alex-jackrabbot\toy\0\static\ped_file.csv';
        %
        % mapname = 'C:\Users\Alexandre\Dropbox\Stages\Stanford\alex-jackrabbot\toy\0\static\sidewalk.mat';
        %
        %
        %
        %
        % Pedstart = importpedstart(filename);
        %[dest,Path,LocalGoal,Obst] = mapanalysis(mapname,0);
        
        %[dest,Path,LocalGoal,Obst] = mapanalysis('//Users//alex//Dropbox//Jackrabbot-alex//toy//0//static//sidewalk.mat');
        
        
        
        lt = 50;    % time length during when people can get in
        nped = 20;  % number of agent
        
        %random simulation of the number of agent and frequence of apparition.
        r = rand(1,lt);
        r = round(r.*nped./sum(r));
        
        ndest = size(dest,1);
        cen = mean(dest); % barycenter point of the map
        Vm = repmat(cen,ndest,1)-dest;
        Vm = Vm./norm(Vm);
        
        TrS = []; N=0; Nind =0; Pedrem = [];t=0; Pi=[]; dl={}; DepGoal=[];
        
        vm = .5;
        %
        aviobj = VideoWriter('simtrajEWAPtoy2.avi');
        % open(aviobj);
        
        % reevaluation localgoal (not too close to the obstacles)
        % lg=[];
        % for k=1:length(LocalGoal)
        %     for i=1:size(LocalGoal{k},1)
        %         [a,b] = min(sum(sqrt((repmat(LocalGoal{k}(i,:),size(Obst,1),1)-Obst).^2),2));
        %         if a<1
        %             LocalGoal{k}(i,:) = LocalGoal{k}(i,:)+...
        %                 10.*((LocalGoal{k}(i,:)-Obst(b,:)));
        %         end
        %     lg=[lg;LocalGoal{k}(i,:) Path(k,2:3)];
        %     end
        % end
        
        
        while N>0 || t<lt
            
            t=t+1;
            
            if t<=lt
                
                npedi = r(t);
                % npedi = size(Pedstart(Pedstart(:,3)==t,:),1);% number of pedestrian gettting in
                N = N+npedi;% number of pedestrian total in the map
                
                Ti=[];
                
                d1 = randi(ndest,1,npedi);
                d2 = randi(ndest,1,npedi);
                %           d2 = mod(d1+1,2);
                %           d2(d2==0)=4;
                
                
                %         d1 = Pedstart(Pedstart(:,3)==t,1)+1;% initial position
                %         d2 = Pedstart(Pedstart(:,3)==t,2)+1;% destination
                
                % if we consider that an agent can't leave from where he entered
                d2 = mod(d2 + ((d1-d2)==0),ndest+1);%+(d2==ndest);
                d2 = d2 + (d2==0);
                
                Ti(1:npedi,3) = t*ones(npedi,1);   % time of the apparition
                
                
                % Ti(1:npedi,3) = Pedstart(Pedstart(:,3)==t,3);     % time of the apparition
                Ti(1:npedi,1:2) = dest(d1,:)+[5*randn(npedi,1) zeros(npedi,1)  ]; % initial position + noise
                Ti(1:npedi,4) = (1:npedi)'+Nind;  % label of the agent
                
                % cell of local goal per agent
                %         Pi=[];
                %         for k=1:length(d1)
                %             Pi = [Pi; Path(Path(:,2)==d1(k)&Path(:,3)==d2(k),1)];
                %             destloc{k+Nind} = LocalGoal{Path(Path(:,2)==d1(k)&Path(:,3)==d2(k),1)};
                %         end
                
                % initial speed of entrance
                Vi = Vm(d1,:); + 0.005*randn(npedi,2); %#ok<VUNUS>
                % Vi = Pedstart(Pedstart(:,3)==t,4);
                
                % array containing DepGoal(label agent, origine, destination, path)
                % DepGoal = [DepGoal; (1+Nind:npedi+Nind)' d1' d2' Pi];
                
                DepGoal = [DepGoal; (1+Nind:npedi+Nind)' d1' d2' ];
                
                % TrS(id,px,py,vx,vy,pdestx,pdesty,u,gid,time,path)
                if npedi>0
                    TrS = [TrS;...
                        Ti(:,[4 1 2]) ...    % label + px + py
                        Vi ...               % initial speed
                        dest(d2,:)+[5*randn(npedi,1) zeros(npedi,1)  ] ...       % final destination
                        vm*ones(npedi,1)+rand(npedi,1)...
                        ones(npedi,1)...
                        t.*ones(npedi,1)];...     % time
                        %                 Pedstart(Pedstart(:,3)==t,4)...  % likely average speed
                    %                 Pedstart(Pedstart(:,3)==t,5)...    % group
                    %                DepGoal(1+Nind:npedi+Nind,4)...%#ok<*AGROW>  % path
                    %                ];
                end
            end
            
            
            % T(id,px,py,vx,vy,pdestx,pdesty,u,gid)
            
            
            if ~isempty(TrS)
                
                % withdraw agents gone.
                T = TrS(TrS(:,end)==t,:);
                if ~isempty(Pedrem)
                    for i=Pedrem'
                        T(T(:,1)==i,:)=[];
                    end
                end
                
                
                % D(id,px,py,vx,vy)
                if ~isempty(T)
                    D = pathPredict(T(:,1:end-1),[],params{1},1,'ewap');
                    
                    D(1:N,:)=[];
                    
                    
                    
                    TrS = [TrS;D T(T(:,1)==D(:,1),[6 7])...
                        T(T(:,1)==D(:,1),8) ones(size(D,1),1) ...
                        (t+1)*ones(size(D,1),1)]; %#ok<*AGROW>
                    
                    
                    
                    
                    % if someone is too close from its goal, or out of the map, errase him
                    Tnew = TrS(TrS(:,end)==(t+1),1:end);
                    pedsel = ((sum(abs(Tnew(:,2:3)-Tnew(:,6:7)),2))<5)+...
                        (Tnew(:,2)<min(dest(:,1))-10)+(Tnew(:,2)>max(dest(:,1))+10)+...
                        (Tnew(:,3)<min(dest(:,2))-10)+(Tnew(:,3)>max(dest(:,2))+10);
                    
                    Pedrem = unique(Tnew(logical(pedsel),1)); % agent label to remove.
                    
                end
                
                % withdraw the agent out. N = total agent currently in the map.
                N=N-length(Pedrem);
                Nind=Nind+npedi; % number of indice assigned;
                
                
                
                %% Plot & video
                
%                         figure(2)
%                         Vid=TrS(TrS(:,end)==t,:);
%                         plot(Vid((Vid(:,6)==50),2),Vid((Vid(:,6)==50),3),'o');
%                         hold on
%                         plot(Vid((Vid(:,6)==0),2),Vid((Vid(:,6)==0),3),'ro');% current agents' position
%                 
%                        % plot(Obst(:,1),Obst(:,2),'k.')      % map of Obstacles
%                         plot(cen(1),cen(2),'r*')            % barycenter of destinations
%                         plot(dest(:,1),dest(:,2),'b*')
%                         axis([-5 55 -5 55]);% final destinations & entrances
%                         %plot(lg(:,1),lg(:,2),'g*')          % local goals
%                         hold off
%                         frame = getframe;
%                         writeVideo(aviobj,frame);
                
                
                
            end
        end
        
        
        %% saving data.
        
        
%         simulationSF = TrS(:,[2 3 10 1 9]);
%       id, x, y, t 
        simulationSF = TrS(:,[1 2 3 10]);
        %dir = [file '/dynamic/'];
        cd('/Users/Amir/Desktop/Silvio/capri4/eccv_alex/SocialForces')
        csvwrite(['simulationRdSF' num2str(w) '.csv'],simulationSF);
        
    end
end
%% Final plot of the trajectories
%
% plot(dest(:,1),dest(:,2),'r*')
% hold on
% for k=1:200
%     p=TrS(TrS(:,1)==k,:);
%     plot(p(:,2),p(:,3));
% %     pause
%     hold on
% end

plot(dest(:,1),dest(:,2),'r*')
hold on
for t=1:200
    p=TrS(TrS(:,10)==t,:)
    scatter(p(:,2),p(:,3));
    pause
    hold on
end


