clear all
close all
load Stanford_UAV_21v.mat
load D_Uav
data = D(21);
for j=400:500
    
    try %#ok<*TRYNC>
        cd /scr/alexr/SocialForces/
        fprintf([num2str(j) '\n']);
        clear TrS Data D Ti Pedrem N Nind dloc npedi LocalGoal DepGoal PedStart Path Obst
        addpath('FastMarching_version3b','SFfunctions')
        
        params = {[1.34086 1.068818 0.852700 0.0840671 .00999 0.442588 0.0],...
            [1.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420]};
        method='ewap';
        
        ratio = 1.5;
        
        Is = data.mask./9;
        Im = data.fframeH;
        dest = data.dest./ratio;
        path = data.path;
        LocalGoal = data.lg;
        c = data.c;
        
        Obst=[];
        
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
        vm = .45;
        
        while N>0 && t<1000 || t<lt
            
            t=t+1;
            npedi=0;
            if t<=lt
                npedi = r(t);
                if npedi>0
                    % npedi = size(Pedstart(Pedstart(:,3)==t,:),1);% number of pedestrian gettting in
                    N = N+npedi;% number of pedestrian total in the map
                    
                    Ti=[];
                    
                    % path, trajectory and class
                    
                    p1 = path(randi(length(path),npedi,1),:);
                    c1 = c(p1(:,1),:); %class(0 for bike and 1 for ped 2 for both)
                    d1 = p1(:,2); d2 = p1(:,3);
                    
                    
                    
                    % cell of local goal per agent
                    for k=1:size(p1,1) %#ok<*ALIGN>
                        if c1(k) == 2
                            c1(k) = randi(2)-1;
                            destloc{k+Nind} = LocalGoal{p1(k,1)}{c1+1}./ratio;
                        else
                            destloc{k+Nind} = LocalGoal{p1(k,1)}./ratio;
                        end
                    end
                    
                    Ti(1:npedi,3) = t*ones(npedi,1);   % time of the apparition
                    
                    
                    % Ti(1:npedi,3) = Pedstart(Pedstart(:,3)==t,3);     % time of the apparition
                    Ti(1:npedi,1:2) = dest(d1,:);%+randn(npedi,2) ;% initial position + noise
                    % Ti(1:npedi,1:2) = dest(d1,:)+[5*randn(npedi,1) zeros(npedi,1) ];
                    Ti(1:npedi,4) = (1:npedi)'+Nind;  % label of the agent
                    
                    
                    % initial speed of entrance
                    Vi = .01*Vm(d1,:) + 0.1*randn(npedi,2);
                    
                    
                    % array containing DepGoal(label agent, origine, destination, path)
                    % DepGoal = [DepGoal; (1+Nind:npedi+Nind)' d1' d2' Pi];
                    
                    DepGoal = [DepGoal; (1+Nind:npedi+Nind)' p1(:,[2 3 1]) ]; %#ok<*AGROW>
                    
                    % TrS(id,px,py,vx,vy,pdestx,pdesty,u,gid,time,path)
                    
                    TrS = [TrS;...
                        Ti(:,[4 1 2]) ...    % label + px + py
                        Vi ...               % initial speed
                        dest(d2,:) ...       % final destination
                        vm*ones(npedi,1)-(2*vm/3)*c1+0.000005.*rand(npedi,1)...% likely average speed
                        ones(npedi,1)...% group
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
                            if sum(abs(D(D(:,1)==i,2:3)-dl{i}))<3 %&& dl{i}~= dest(DepGoal(i,3),:)
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
                        (t+1)*ones(size(D,1),1) ...
                        T(T(:,1)==D(:,1),end)];
                    
                    
                    
                    % if someone is too close from its goal, or out of the map, errase him
                    Tnew = TrS(TrS(:,end-1)==(t+1),1:end);
                    pedsel = ((sum(abs(Tnew(:,2:3)-Tnew(:,6:7)),2))<1)+...
                        (Tnew(:,2)<min(dest(:,1))-10)+(Tnew(:,2)>max(dest(:,1))+10)+...
                        (Tnew(:,3)<min(dest(:,2))-10)+(Tnew(:,3)>max(dest(:,2))+10);
                    
                    Pedrem = unique(Tnew(logical(pedsel),1)); % agent label to remove.
                    
                end
                
                % withdraw the agent out. N = total agent currently in the map.
                N=N-length(Pedrem);
                Nind=Nind+npedi; % number of indice assigned;
                
                
                
                %% Plot & video
                % figure(2)
%                 imshow(Im)
%                 hold on
%                 Vid=TrS(TrS(:,end-1)==t,:);
%                 plot(ratio*Vid(:,2),ratio*Vid(:,3),'co','LineWidth',2);        % current agents' position             plot(Obst(:,1),Obst(:,2),'k.')      % map of Obstacles
%                 %plot(cen(1),cen(2),'r*')            % barycenter of destinations
%                 plot(ratio*dest(:,1),ratio*dest(:,2),'b*')      % final destinations & entrances
%                 %plot(lg(:,1),lg(:,2),'g*')          % local goals
%                 hold off
%                 pause(0.001)     %                 %             frame = getframe;
                %   writeVideo(aviobj,frame);
                %
                %
                %
            end
        end
        
        
        %% saving data.
        simulationSF = TrS(:,[2 3 10 1 9]);
        Crop=[];
        fprintf('cropping')
        for k = 1:size(simulationSF,1)
            pos = simulationSF(k,1:2);
            rect = [pos(1)-16 pos(2)-16 32 32];
            crop = imcrop(Im,rect);
            %size(crop)
            %         figure(1)
            %         imshow(crop)
            %         figure(2)
            %         imshow(Is)
            %         hold on
            %         plot(pos(1),pos(2),'*');
            %         hold off
%             c = zeros(33,33,3);
%             for ic1=1:33
%                 for ic2=1:33
%                     for ic3=1:3 %#ok<*FXSET>
%                         try
%                             c(ic1,ic2,ic3)=crop(ic1,ic2,ic3);
%                         end
%                     end
%                 end
%             end
            if size(crop,1)<33
                crop = [crop ;  zeros(33-size(crop,1),size(crop,2),3)];
            end
            if size(crop,2)<33
                crop = [crop  zeros(33,33-size(crop,2),3)];
            end
           
            Crop=[Crop;crop(:)'];
        end
        simulationSF(:,[1 2])=round(simulationSF(:,[1 2])); %#ok<*NASGU>
        % Data(px,py,time,id,group)
        %Data = TrS([2 3 4 1 9],:);
        simulationSF = TrS(:,[2 3 10 1 9]);
        simulationSF(:,[1 2])=round(simulationSF(:,[1 2]));
        % cd Destinations\
        % csvwrite(['dest' num2str(f) '.csv'],dest);
        % cd ..
        cd /scr/alexr/simtrajRdSF_Eccv5_2/
        csvwrite(['simtrajRdSF' num2str(j) '.csv'],simulationSF);
        save(['crop_' num2str(j)],'Crop')
        %save simtrajEWAP5 TrS
    end
end