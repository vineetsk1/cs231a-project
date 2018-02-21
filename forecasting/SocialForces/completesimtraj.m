
clear all
close all

%%
addpath('FastMarching_version3b','SFfunctions')
for f=89:499
    cd /scr/alexr/SocialForces/
    fprintf([num2str(f) '\n'])
    clear TrS Data D Ti Pedrem N Nind dloc npedi LocalGoal DepGoal PedStart Path Obst r
    %dest=[0 10;0 30;50 30;50 10];
    %cd /scr/alexr/SocialForces/
    %dest = [0 25;50 25];
    %dest = [25 0;25 50];
    params = {[.534086 2.68818 0.0822700 200.940671 .505634 0.842588],...
        [1.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420]};
    method='ewap';
    try %#ok<TRYNC>
        % filename = 'C:\Users\Alexandre\Dropbox\Stages\Stanford\alex-jackrabbot\toy\0\static\ped_file.csv';
        %
        file = ['/scratch/alexz/from_local/toy_dataset_1/' num2str(f)];
        
        filename = [file '/static/ped_file.csv'];
        
        mapname = [file '/static/sidewalk.mat'];
        
        [dest,Path,LocalGoal,Obst] = mapanalysis(mapname,0);
        
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
        % aviobj = VideoWriter('simtrajEWAPtoy2.avi');
        % open(aviobj);
        
        % reevaluation localgoal (not too close to the obstacles)
        lg=[];
        for k=1:length(LocalGoal)
            for i=1:size(LocalGoal{k},1)
                [a,b] = min(sum(sqrt((repmat(LocalGoal{k}(i,:),size(Obst,1),1)-Obst).^2),2));
                if a<1
                    LocalGoal{k}(i,:) = LocalGoal{k}(i,:)+...
                        10.*((LocalGoal{k}(i,:)-Obst(b,:)));
                end
                lg=[lg;LocalGoal{k}(i,:) Path(k,1)];
            end
        end
        
        npath = length(Path);
        Pi=[];
        while N>0 || t<lt
            
            t=t+1;
            
            if t<=lt
                
                npedi = r(t);
                % npedi = size(Pedstart(Pedstart(:,3)==t,:),1);% number of pedestrian gettting in
                N = N+npedi;% number of pedestrian total in the map
                
                Ti=[];
                
                pi = randi(npath,1,npedi);
                d1 = Path(pi,2); d2 = Path(pi,3);
                
                
                Ti(1:npedi,3) = t*ones(npedi,1);   % time of the apparition
                
                
                % Ti(1:npedi,3) = Pedstart(Pedstart(:,3)==t,3);     % time of the apparition
                Ti(1:npedi,1:2) = dest(d1,:)+[5*randn(npedi,1) zeros(npedi,1)  ]; % initial position + noise
                Ti(1:npedi,4) = (1:npedi)'+Nind;  % label of the agent
                
                % cell of local goal per agent
                Pi = [Pi;pi'];
                for k=1:length(d1)
                    destloc{k+Nind} = cen;
                    %destloc{k+Nind} = lg(lg(:,3)==Pi(k,1),1:2);
                end
                
                % initial speed of entrance
                Vi = Vm(d1,:); + 0.005*randn(npedi,2); %#ok<VUNUS>
                
                
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
                    
                    dloc=[];
                    for i=D(:,1)'
                        
                        % if there is no local goal anymore, head to the final
                        % destination.
                        if isempty(destloc{i})
                            dl{i}= dest(DepGoal(i,3),:);
                            
                            % else, either keep heading to the local goal, either
                            % already to close and then consider the next one.
                        else
                            dl{i} = destloc{i}(1,:); %#ok<*SAGROW>
                            if sum(abs(D(D(:,1)==i,2:3)-dl{i}))<30 %&& dl{i}~= dest(DepGoal(i,3),:)
                                destloc{i}(1,:)=[];
                                if isempty(destloc{i})
                                    dl{i}= dest(DepGoal(i,3),:);
                                else
                                    dl{i} = destloc{i}(1,:);
                                end
                            end
                        end
                        dloc=[dloc;dl{i}];
                    end
                    
                    TrS = [TrS;D dloc...
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
                %
                %         figure(2)
                %             Vid=TrS(TrS(:,end)==t,:);
                %             plot(Vid(:,2),Vid(:,3),'o');
                %
                %             %                     plot(Vid((Vid(:,6)==50),2),Vid((Vid(:,6)==50),3),'o');
                %             %         hold on
                %             %         plot(Vid((Vid(:,6)==0),2),Vid((Vid(:,6)==0),3),'ro');% current agents' position
                %             %
                %             %        % plot(Obst(:,1),Obst(:,2),'k.')      % map of Obstacles
                %             %         plot(cen(1),cen(2),'r*')            % barycenter of destinations
                %             %         plot(dest(:,1),dest(:,2),'b*')
                %             axis([-5 400 -5 400]);% final destinations & entrances
                %             pause
                %         %plot(lg(:,1),lg(:,2),'g*')          % local goals
                %         hold off
                %         frame = getframe;
                %         writeVideo(aviobj,frame);
                %
                %
                %
            end
        end
        
        
        %% saving data.
        
        
        simulationSF = TrS(:,[2 3 10 1 9]);
        %dir = [file '/dynamic/'];
        cd('/scr/alexr/simtrajMapSF/')
        csvwrite(['simulationMapSF' num2str(f) '.csv'],simulationSF);
        
    end
end
%% Final plot of the trajectories
%
% plot(dest(:,1),dest(:,2),'r*')
% hold on
% for k=1:200
%     p=TrS(TrS(:,1)==k,:);
%     plot(p(:,2),p(:,3));
%     hold on
% end
%

