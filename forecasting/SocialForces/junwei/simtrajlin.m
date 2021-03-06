clear all
close all

nped=20;

for j=1:500
    clc
    try %#ok<*TRYNC>
        cd /Users/Amir/Desktop/Silvio/capri4/eccv_alex/SocialForces
        fprintf([num2str(j) '\n']);
        clear TrS Data D Ti Pedrem N Nind dloc npedi LocalGoal DepGoal PedStart Path Obst
        addpath('FastMarching_version3b','SFfunctions')
        
        %if map=='h'
            dest = [ 0 20; 20 20];
            k =1;
        %elseif map=='v'
        %    dest = [ 20 0; 20 40];
        %    k =0;
        %end
        %dest=[0 10;0 30;50 30;50 10];
        
        params = {[1.34086 1.068818 0.852700 0.0840671 .00999 0.442588 0.0],...
            [1.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420]};
        
        
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
        
        
        
        lt = 70;    % time length during when people can get in
        %nped = 20;  % number of agent
        
        %random simulation of the number of agent and frequence of apparition.
        r = rand(1,lt);
        r = round(r.*nped./sum(r));
        
        ndest = size(dest,1);
        cen = mean(dest); % barycenter point of the map
        Vm = repmat(cen,ndest,1)-dest;
        Vm = Vm./norm(Vm);
        
        TrS = []; N=0; Nind =0; Pedrem = [];t=0;  DepGoal=[];
        
        vm = .1;
        %
        %     aviobj = VideoWriter('simtrajEWAPtoy3.avi');
        %     open(aviobj);

        
        while N>0 && t<1000 || t<lt
            
            t=t+1;
            npedi=0;
            if t<=lt
                
                npedi = r(t);
                if npedi>0
                    % npedi = size(Pedstart(Pedstart(:,3)==t,:),1);% number of pedestrian gettting in
                    N = N+npedi;% number of pedestrian total in the map
                    
                    Ti=[];
                    
                    d1 = randi(ndest,1,npedi)
                    d2 = randi(ndest,1,npedi)
                    %           d2 = mod(d1+1,2);
                    %           d2(d2==0)=4;
                    
                    
                    %         d1 = Pedstart(Pedstart(:,3)==t,1)+1;% initial position
                    %         d2 = Pedstart(Pedstart(:,3)==t,2)+1;% destination
                    
                    % if we consider that an agent can't leave from where he entered
                    d2 = mod(d2 + ((d1-d2)==0),ndest+1);%+(d2==ndest);
                    d2 = d2 + (d2==0)
                    
                    Ti(1:npedi,3) = t*ones(npedi,1);   % time of the apparition
                    Ti(1:npedi,1:2) = dest(d1,:);%+[5*(1-k)*randn(npedi,1) 5*k*randn(npedi,1) ];% initial position + noise
                    Ti(1:npedi,4) = (1:npedi)'+Nind;  % label of the agent
                    
                    
                    Vinit = .01*Vm(d1,:);% + 0.1*randn(npedi,2);
                    
                    
                    % array containing DepGoal(label agent, origine, destination, path)
                    % DepGoal = [DepGoal; (1+Nind:npedi+Nind)' d1' d2' Pi];
                    
                    DepGoal = [DepGoal; (1+Nind:npedi+Nind)' d1' d2' ]; %#ok<*AGROW>
                    
                    % TrS(id,px,py,vx,vy,pdestx,pdesty,u,gid,time,path)
                    
                    TrS = [TrS; Ti(:,[4 1 2]) ...    % label + px + py
                        Vinit...               % initial speed
                        dest(d2,:)...+[5*(1-k)*randn(npedi,1) 5*k*randn(npedi,1) ] ...       % final destination
                        vm*ones(npedi,1)...+0.05.*rand(npedi,1)...
                        ones(npedi,1)...
                        t.*ones(npedi,1)];    % time
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
                    % Dpast = D;
                    
                    D(1:N,:)=[];                
                    
                    TrS = [TrS;D T(T(:,1)==D(:,1),[6 7])...
                        T(T(:,1)==D(:,1),8) ones(size(D,1),1) ...
                        (t+1)*ones(size(D,1),1)];
                    %T(T(:,1)==D(:,1),end)];
                    
                    
                    
                    % if someone is too close from its goal, or out of the map, errase him
                    Tnew = TrS(TrS(:,end)==(t+1),1:end);
                    pedsel = ((sum(abs(Tnew(:,2:3)-Tnew(:,6:7)),2))<1)+...
                        (Tnew(:,2)<min(dest(:,1))-10)+(Tnew(:,2)>max(dest(:,1))+10)+...
                        (Tnew(:,3)<min(dest(:,2))-10)+(Tnew(:,3)>max(dest(:,2))+10);
                    
                    Pedrem = unique(Tnew(logical(pedsel),1)); % agent label to remove.
                    
                end
                
                % withdraw the agent out. N = total agent currently in the map.
                N=N-length(Pedrem);
                Nind=Nind+npedi; % number of indice assigned;
                
                
                
                %% Plot & video
                %             figure(2)
                            Vid=TrS(TrS(:,end)==t,:);
                            plot(Vid(:,2),Vid(:,3),'o');        % current agents' position
                            axis([-5 45 5 35]);pause
                %             hold on
                %
                %
                %             % plot(Obst(:,1),Obst(:,2),'k.')      % map of Obstacles
                %             plot(cen(1),cen(2),'r*')            % barycenter of destinations
                %             plot(dest(:,1),dest(:,2),'b*')      % final destinations & entrances
                %             % plot(lg(:,1),lg(:,2),'g*')          % local goals
                %             hold off
                             
            end
        end
        
        
        %% saving data.
        
        % Data(px,py,time,id,group)
        %Data = TrS([2 3 4 1 9],:);
        simulationSF = TrS(:,[2 3 10 1 9]);
        simulationSF(:,[1 2])=round(10.*simulationSF(:,[1 2]));
        % cd Destinations\
        % csvwrite(['dest' num2str(f) '.csv'],dest);
        % cd ..
%         cd /scr/alexr/simtrajLinSF/
%         csvwrite(['simtrajLinSF' num2str(j) '.csv'],simulationSF);
        %save simtrajEWAP5 TrS
    end
end
