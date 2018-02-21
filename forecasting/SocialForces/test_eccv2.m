
clear all
close all

for j=312:400
    
    try%#ok<*TRYNC>
        f=j;
        cd /scr/alexr/SocialForces/
        fprintf([num2str(j) '\n']);
        clear TrS Data D Ti Pedrem N Nind dloc npedi LocalGoal DepGoal PedStart Path Obst
        addpath('FastMarching_version3b','SFfunctions')
        %dest=[0 10;0 30;50 30;50 10];
        dest = [ 0 20; 20 0; 40 20;20 40];
        %dest = [ 25 0; 25 10];
        params = {[1.34086 1.068818 0.852700 0.0840671 .00999 0.442588 0.0],...
            [1.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420]};
        method='ewap';
        file = ['/scratch/alexz/from_local/toy_dataset_1/' num2str(j)];
        
        filename = [file '/static/ped_file.csv'];
        
        mapname = [file '/static/sidewalk.mat'];
        % filename = 'C:\Users\Alexandre\Dropbox\Stages\Stanford\alex-jackrabbot\toy\0\static\ped_file.csv';
        %
        % mapname = 'C:\Users\Alexandre\Dropbox\Stages\Stanford\alex-jackrabbot\toy\0\static\sidewalk.mat';
        %
        %
        %
        %
        Pedstart = importpedstart(filename);
        
        
        Is = load(mapname);
        Is = Is.arr;
        [dest,Path,LocalGoal,Obst] = mapanalysis(mapname,0);
        
        %[dest,Path,LocalGoal,Obst] = mapanalysis('//Users//alex//Dropbox//Jackrabbot-alex//toy//0//static//sidewalk.mat');
        
        
        
        lt = 50;    % time length during when people can get in
        nped = 100;  % number of agent
        
        %random simulation of the number of agent and frequence of apparition.
        r = rand(1,lt);
        r = round(r.*nped./sum(r));
        
        ndest = size(dest,1);
        [~,dm]=min(dest);[~,dM]=max(dest);
        cen = [mean(dest([dm(2) dM(2)],1)) mean(dest([dm(1) dM(1)],2))];
        %cen = median(dest); % barycenter point of the map
        Vm = repmat(cen,ndest,1)-dest;
        Vm = Vm./norm(Vm);
        
        TrS = []; N=0; Nind =0; Pedrem = [];t=0; Pi=[]; dl={}; DepGoal=[];
        
        vm = .2;
        %
        %     aviobj = VideoWriter('simtrajEWAPtoy3.avi');
        %     open(aviobj);
        
        % reevaluation localgoal (not too close to the obstacles)
        lg=[];
        for k=1:length(LocalGoal)
            for i=1:size(LocalGoal{k},1)
                [a,b] = min(sum(sqrt((repmat(LocalGoal{k}(i,:),size(Obst,1),1)-Obst).^2),2));
                if a<1
                    LocalGoal{k}(i,:) = LocalGoal{k}(i,:)+...
                        30.*((LocalGoal{k}(i,:)-Obst(b,:)));
                end
                lg=[lg;LocalGoal{k}(i,:) Path(k,2:3)];
            end
        end
        
        
        
        
        
        method='ewap';
        
        % [dest,Path,LocalGoal,Obst] = mapanalysis(mapname,0);
        
        lt = 50;    % time length during when people can get in
        
        
        %     nped = 20;  % number of agent
        %
        %     %random simulation of the number of agent and frequence of apparition.
        %     r = rand(1,lt);
        %     r = round(r.*nped./sum(r));
        
        ndest = size(dest,1);
       % cen = mean(dest); % barycenter point of the map
        Vm = repmat(cen,ndest,1)-dest;
        Vm = Vm./norm(Vm);
        
        TrS = []; N=0; Nind =0; Pedrem = [];t=0; Pi=[]; dl={}; DepGoal=[];
        
        vm = 5;
        
        % aviobj = VideoWriter('simtrajEWAP5.avi');
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
                lg=[lg;LocalGoal{k}(i,:) Path(k,2:3)];
            end
        end
        
        
        
        
        
        while N>0 || t<lt
            
            
            t=t+1;
            
            if t<=lt
                %   npedi = r(t);
                npedi = size(Pedstart(Pedstart(:,3)==t,:),1);% number of pedestrian gettting in
                N = N+npedi;% number of pedestrian total in the map
                
                Ti=[];
                
                %   d1 = randi(ndest,1,npedi);
                d1 = Pedstart(Pedstart(:,3)==t,1)+1;% initial position
                %   d2 = randi(ndest,1,npedi);
                d2 = Pedstart(Pedstart(:,3)==t,2)+1;% destination
                
                %         % if we consider that an agent can't leave from where he entered
                %         d2 = mod(d2 + ((d1-d2)==0),ndest+1);%+(d2==ndest);
                %         d2 = d2 + (d2==0);
                
                %         Ti(1:npedi,3) = t*ones(npedi,1);   % time of the apparition
                
                
                Ti(1:npedi,3) = Pedstart(Pedstart(:,3)==t,3);     % time of the apparition
                Ti(1:npedi,1:2) = dest(d1,:)+0.1*randn(npedi,2); % initial position + noise
                Ti(1:npedi,4) = (1:npedi)'+Nind;  % label of the agent
                
                % cell of local goal per agent
                Pi=[];
                for k=1:length(d1)
                    Pi = [Pi; Path(Path(:,2)==d1(k)&Path(:,3)==d2(k),1)];
                    destloc{k+Nind} =cen;% LocalGoal{Path(Path(:,2)==d1(k)&Path(:,3)==d2(k),1)};
                end
                
                % initial speed of entrance
                Vi = Vm(d1,:); + 0.005*randn(npedi,2); %#ok<VUNUS>
                % Vi = Pedstart(Pedstart(:,3)==t,4);
                
                % array containing DepGoal(label agent, origine, destination, path)
                DepGoal = [DepGoal; (1+Nind:npedi+Nind)' d1' d2' Pi];
                
                % TrS(id,px,py,vx,vy,pdestx,pdesty,u,gid,time,path)
                if npedi>0
                    TrS = [TrS;...
                        Ti(:,[4 1 2]) ...    % label + px + py
                        Vi ...               % initial speed
                        dest(d2,:) ...       % final destination
                        Pedstart(Pedstart(:,3)==t,4)...  % likely average speed
                        Pedstart(Pedstart(:,3)==t,5)...    % group
                        t.*ones(npedi,1)...     % time
                        DepGoal(1+Nind:npedi+Nind,4)];  %#ok<*AGROW>  % path
                end
            end
            
            
            % T(id,px,py,vx,vy,pdestx,pdesty,u,gid)
            
            
            if ~isempty(TrS)
                
                % withdraw agents gone.
                T = TrS(TrS(:,end-1)==t,:);
                if ~isempty(Pedrem)
                    for i=Pedrem'
                        T(T(:,1)==i,:)=[];
                    end
                end
                
                
                % D(id,px,py,vx,vy)
                if ~isempty(T)
                    D = pathPredict(T(:,1:end-2),Obst,params{1},1,'ewap');
                    
                    D(1:N,:)=[];
                    
                    % local destination actualization
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
                            if sum(abs(D(D(:,1)==i,2:3)-dl{i}))<40 %&& dl{i}~= dest(DepGoal(i,3),:)
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
                    
                    % the array is then actulaize by considering the local
                    % trajectory identify.
                    TrS = [TrS;D dloc...
                        T(T(:,1)==D(:,1),8) ones(size(D,1),1) ...
                        (t+1)*ones(size(D,1),1) ...
                        T(T(:,1)==D(:,1),end)];
                    
                    
                    
                    % if someone is too close from its goal, or out of the map, errase him
                    Tnew = TrS(TrS(:,end-1)==(t+1),1:end-1);
                    pedsel = ((sum(abs(Tnew(:,2:3)-Tnew(:,6:7)),2))<10)+...
                        (Tnew(:,2)<min(dest(:,1))-10)+(Tnew(:,2)>max(dest(:,1))+10)+...
                        (Tnew(:,3)<min(dest(:,2))-10)+(Tnew(:,3)>max(dest(:,2))+10);
                    
                    Pedrem = unique(Tnew(logical(pedsel),1)); % agent label to remove.
                    
                end
                
                % withdraw the agent out. N = total agent currently in the map.
                N=N-length(Pedrem);
                Nind=Nind+npedi; % number of indice assigned;
                
                
                
                %% Plot & video
%                 
%                                          figure(2)
%                                          Vid=TrS(TrS(:,end-1)==t,:);
%                                          plot(Vid(:,2),Vid(:,3),'o');        % current agents' position
%                                          hold on
%                                          plot(Obst(:,1),Obst(:,2),'k.')      % map of Obstacles
%                                          plot(cen(1),cen(2),'r*')            % barycenter of destinations
%                                          plot(dest(:,1),dest(:,2),'b*')      % final destinations & entrances
%                                          plot(lg(:,1),lg(:,2),'g*')          % local goals
%                                          hold off
%                                          pause(0.001)
                %                 %         frame = getframe;
                %writeVideo(aviobj,frame);
                %
                %
                %
            end
        end
        
        %% saving data.
        
        % Data(px,py,time,id,group)
        %Data = TrS([2 3 4 1 9],:);
        simulationSF = TrS(:,[2 3 10 1 9]);
        Crop=[];
        for k = 1:size(simulationSF,1)
            pos = simulationSF(k,1:2);
            rect = [pos(1)-16 pos(2)-16 32 32];
            crop = imcrop(Is,rect);
            %size(crop)
            %         figure(1)
            %         imshow(crop)
            %         figure(2)
            %         imshow(Is)
            %         hold on
            %         plot(pos(1),pos(2),'*');
            %         hold off
            c = zeros(1,33*33);
            crop = crop(:);
            c(1:length(crop))=crop;
            Crop = [Crop;c];
        end
        simulationSF(:,[1 2])=round(simulationSF(:,[1 2]));
        % cd Destinations\
        % csvwrite(['dest' num2str(f) '.csv'],dest);
        % cd ..
        cd /scr/alexr/simtrajRdSF_clean_Eccv2/
        csvwrite(['simtrajRdSF' num2str(j) '.csv'],simulationSF);
        %csvwrite(['cop' num2str(j) '.csv'],Crop);
        save(['crop_' num2str(j)],'Crop')
        %save simtrajEWAP5 TrS
    end
end
