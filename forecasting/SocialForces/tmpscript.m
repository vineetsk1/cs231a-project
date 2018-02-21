% % Temporally script
% 
% % Tracking result visualization
% clear;
% F3 = load('main9resultF3.mat');
% F4 = load('main9resultF4.mat');
% R = [F4.R;F3.R];
% S = [F4.S F3.S];
% config = F3.config;
% clear F3 F4;
% 
% enable = [5 6 7];
% R = R(:,enable);
% config = config(enable,:);
% 
% 
% % Annotation override
% for j = 1:size(config,1)
%     if strcmp(config{j,1},'ewap'), config{j,1} = 'lta'; end
%     if strcmp(config{j,2},'mix'), config{j,2} = 'full'; end
%     config{j,1} = upper(config{j,1});
%     config{j,2} = upper(config{j,2});
% end
% % 
% 
% % Video
% appRate = 4;
% tmplSize = [32 16];
% anchorPos = [-24 0];
% c = fliplr(hsv(size(config,1)));
% for Dataset = {'zara01','zara02','students03'}
%     fprintf('%s\n',Dataset{:});
%     ind = arrayfun(@(s) strcmp(s.dataset,Dataset{:}),S);
%     s = S(ind);
%     r = R(ind,:);
%     
%     %
%     aviobj = avifile(['tracking_less_' Dataset{:} '.avi'],'fps',2.5);
% 
%     % Render
%     for i = 1:length(s)
%         Pprev = [];
%         PIDprev = [];
%         for t = 1:length(s(i).frames)
%             if mod(t,appRate)~=0, continue; end
%             ind = r{i,1}.obsv(:,1)==s(i).frames(t);
%             % Skip if empty
%             if ~any(ind), continue; end
%             
%             % Compute points
%             P = zeros(nnz(ind),2,size(config,1));
%             for j = 1:size(config,1)
%                 p = r{i,j}.obsv(ind,3:4);
%                 p = [p ones(size(p,1),1)] / s(i).H';
%                 P(:,:,j) = [p(:,1)./p(:,3) p(:,2)./p(:,3)];
%             end
%             % Person id
%             PID = r{i,1}.obsv(ind,2);
%             
%             % Show image
%             imshow(read(s(i).vid,s(i).frames(t)));
%             
%             % Show annotation
%             hold on;
%             l = zeros(size(config,1),1);
%             for j = 1:size(config,1)
%                 l(j) = plot(P(:,2,j),P(:,1,j),'*','Color',c(j,:),'MarkerSize',12);
%                 for k = 1:size(P,1)
%                     % Draw line
%                     if ~isempty(PIDprev==PID(k))
%                         plot([Pprev(PIDprev==PID(k),2,j) P(k,2,j)],...
%                              [Pprev(PIDprev==PID(k),1,j) P(k,1,j)],...
%                              'Color',c(j,:),'LineWidth',2);
%                     end
%                     % Draw box
%                     rectangle('Position',...
%                         [P(k,2,j)-tmplSize(2)+anchorPos(2)...
%                          P(k,1,j)-tmplSize(1)+anchorPos(1)...
%                          2*tmplSize(2) 2*tmplSize(1)],...
%                         'EdgeColor',c(j,:),...
%                         'LineWidth',3);
%                 end
%             end
%             text(5,10,sprintf('Sequence %d, Dataset %s, Frame %d',...
%                               i,s(i).dataset,s(i).frames(t)),...
%                 'Color','w','FontSize',12,'FontWeight','bold');
% %             legend(l,strcat(strcat(config(:,1),'+'),config(:,2)),...
% %                 'Location','SouthEast');
%             legend(l,config(:,1),'Location','SouthEast',...
%                 'FontSize',12,'FontWeight','bold');
%             hold off;
%             
%             %
%             aviobj = addframe(aviobj,gca);
%             
%             % Keep history
%             Pprev = P;
%             PIDprev = PID;
%         end
%     end
%     aviobj = close(aviobj);
% end
% 
% 
% 
% % Tracking result analysis
% clear;
% F3 = load('main9resultF3.mat');
% F4 = load('main9resultF4.mat');
% R = [F4.R;F3.R];
% S = [F4.S F3.S];
% config = F3.config;
% clear F3 F4;
% 
% maxDist = 0.5;
% stepRange = [5:4:25; 5:4:25]';% [9:8:25; 9:8:25]';
% Datasets = {'zara01','zara02','students03'};
% fprintf('Maximum Distance: %1.2f (m)\n',maxDist);
% % Drop irrelevant dataset to check
% SUCCESS = cell(length(Datasets),1);
% LOST = cell(length(Datasets),1);
% ID_SWITCH = cell(length(Datasets),1);
% DIST = cell(length(Datasets),1);
% for Dataset = Datasets
%     fprintf('%s\n',Dataset{:});
%     did = strmatch(Dataset,Datasets);
%     SUCCESS{did} = zeros(length(stepRange),size(config,1));
%     LOST{did} = zeros(length(stepRange),size(config,1));
%     ID_SWITCH{did} = zeros(length(stepRange),size(config,1));
%     DIST{did} = zeros(length(stepRange),size(config,1));
%     for j = 1:size(stepRange,1)
%         fprintf(' Tracking Step: %d\n',stepRange(j,1));
%         ind = arrayfun(@(s) strcmp(s.dataset,Dataset{:}),S);
%         s = S(ind);
%         r = R(ind,:);
% 
%         % Analyze
%         L = salvage(s,r,config,'MaxDist',maxDist,'StepRange',stepRange(j,:));
% 
%         % Show
%         SUCCESS{did}(j,:) = sum(cellfun(@(l) nnz(l(:,3)~=0&l(:,2)==l(:,3)), L));
%         LOST{did}(j,:) = sum(cellfun(@(l) nnz(l(:,3)==0), L));
%         ID_SWITCH{did}(j,:) = sum(cellfun(@(l) nnz(l(:,3)~=0&l(:,2)~=l(:,3)), L));
%         DIST{did}(j,:) = mean(cell2mat(cellfun(@(l) l(:,4), L,'UniformOutput',false)));
%         fprintf('  Success:      ');
%         fprintf(' %5d',SUCCESS{did}(j,:)); fprintf('\n');
%         fprintf('  Lost:         ');
%         fprintf(' %5d',LOST{did}(j,:)); fprintf('\n');
%         fprintf('  ID switch:    ');
%         fprintf(' %5d',ID_SWITCH{did}(j,:)); fprintf('\n');
%         fprintf('  Failed total: ');
%         fprintf(' %5d',LOST{did}(j,:)+ID_SWITCH{did}(j,:)); fprintf('\n');
%         fprintf('  Distance:     ');
%         fprintf(' %2.3f',DIST{did}(j,:)); fprintf('\n');
%     end
% end
% save tracking_result_for_barplot.mat;
% 
% % Table output
% load;
% % Change the text label
% for j = 1:size(config,1)
%     if strcmp(config{j,1},'ewap'), config{j,1} = 'lta'; end
%     if strcmp(config{j,2},'mix'), config{j,2} = 'full'; end
% end
% 
% labels = upper(strcat(strcat(config(:,1),'+'),config(:,2)));
% labels{1} = 'CORR';
% labels = char(labels);
% Datasets{3} = 'stu03';
% 
% % For each method
% for j = 1:size(config,1)
%     fprintf('&%s ',labels(j,:));
%     for i = 1:length(Datasets)
%         % For each step
%         for k = 1:size(SUCCESS{i},1)
%             fprintf('& %3d ',LOST{i}(k,j));
%         end
%     end
%     fprintf('\\\\\n');
% end
% 
% % Barplot of tracking result
% load;
% % Change the text label
% for j = 1:size(config,1)
%     if strcmp(config{j,1},'ewap'), config{j,1} = 'lta'; end
%     if strcmp(config{j,2},'mix'), config{j,2} = 'full'; end
% end
% 
% labels = upper(strcat(strcat(config(:,1),'+'),config(:,2)));
% labels{1} = 'CORR';
% Datasets{3} = 'stu03';
% 
% for i = 1:length(Datasets)
%     ind = 1:size(SUCCESS{i},1);%[2 3 4];
%     subplot(1,3,i);
%     h = bar(SUCCESS{i}(ind,:),'group');
% %     title(['Dataset: ' Datasets{i} ' (maximum=' num2str(maxDist) ')']);
%     title(Datasets{i});
%     legend(labels,'FontSize',8);
%     ylabel('Success count');
%     xlabel('Prediction step');
%     ylim([0 max(max(SUCCESS{i}(ind,:)))+100]);
%     set(gca,'XTick',1:length(ind),'XTickLabel',num2str(stepRange(ind,1)-1));
%     
%     % Adds text
%     X = max(cell2mat(get(h,'XData')),[],1);
%     Y = max(cell2mat(get(h,'YData')),[],1);
%     text(X-0.4,Y+20,...
%         strcat('#TRK: ',num2str(SUCCESS{i}(ind,1)+LOST{i}(ind,1)+ID_SWITCH{i}(ind,1))),...
%         'FontSize',8);
%     
% %     pause;
% end
% 
% % Larger barplots of tracking result
% load tracking_result_for_barplot.mat;
% % Change the text label
% for j = 1:size(config,1)
%     if strcmp(config{j,1},'ewap'), config{j,1} = 'lta'; end
%     if strcmp(config{j,2},'mix'), config{j,2} = 'full'; end
% end
% 
% labels = upper(strcat(strcat(config(:,1),'+'),config(:,2)));
% labels{1} = 'CORR';
% Datasets{3} = 'student03';
% 
% for i = 1:length(Datasets)
%     ind = 1:size(SUCCESS{i},1);%[2 3 4];
%     subplot(3,1,i);
%     h = bar(SUCCESS{i}(ind,:),'group');
% %     title(['Dataset: ' Datasets{i} ' (maximum=' num2str(maxDist) ')']);
%     title(Datasets{i});
%     legend(labels,'FontSize',8);
%     ylabel('Success count');
%     xlabel('Prediction step');
%     ylim([0 max(max(SUCCESS{i}(ind,:)))+100]);
%     set(gca,'XTick',1:length(ind),'XTickLabel',num2str(stepRange(ind,1)-1));
%     
%     % Adds text
%     X = max(cell2mat(get(h,'XData')),[],1);
%     Y = max(cell2mat(get(h,'YData')),[],1);
%     text(X-0.4,Y+20,...
%         strcat('#TRK: ',num2str(SUCCESS{i}(ind,1)+LOST{i}(ind,1)+ID_SWITCH{i}(ind,1))),...
%         'FontSize',8);
%     
% %     pause;
% end
% set(gcf,'PaperUnits','inches');
% set(gcf,'PaperSize',[6.5 9]);
% set(gcf,'PaperPositionMode','manual');
% set(gcf,'PaperPosition',[0 0 6.5 9]);
% print('-depsc2','fig_tracking_barplot.eps');
% 
% % Tracking result plot by mesh
% load main9resultF3.mat
% % For each (method,simulation)
% close all;
% Nduration = max(arrayfun(@(s) length(s.frames)-s.offset-1,S));
% edges = [linspace(0,5,11) inf]; % Error for each timesteps
% thresh = [linspace(0,1.5,16) inf]; % Thresholds for successful tracks
% Datasets = 'zara01';    % zara01, zara02, students03
% 
% % Drop irrelevant dataset to check
% ind = arrayfun(@(s) ~strcmp(s.dataset,Datasets),S);
% S(ind) = [];
% R(ind,:) = [];
% 
% TT = zeros(size(config,1),length(thresh));
% err = cell(length(S),length(config));
% for i = 1:size(config,1)
%     % Compute histogram and successful guys
%     H = zeros(length(edges),Nduration);
%     T = cell(length(S),1);
%     for j = 1:length(S)
%         T{j} = false(size(S(j).trks,1),length(thresh));
%         e = cell(size(S(j).trks,1),1);
%         % Check (step,error) for each track
%         for k = 1:size(S(j).trks,1)
%             ind = find(R{j,i}.obsv(:,2) == S(j).trks(k,1) &...  % person
%                        R{j,i}.obsv(:,1) >  S(j).trks(k,2));     % time
%             E = arrayfun(@(k) sqrt(sum((...
%                 S(j).obsv(S(j).obsv(:,1)==R{j,i}.obsv(k,1) &...
%                           S(j).obsv(:,2)==R{j,i}.obsv(k,2),3:4)-...
%                 R{j,i}.obsv(k,3:4)).^2,2)),...
%                 ind);
%             % [1:length(E),E] = [steps,E]
%             % Accumulate count at the corresponding bins
%             for l = 1:length(E)
%                 H(:,l) = H(:,l) + histc(E(l),edges)';
%             end
%             % Check the success
%             for l = 1:length(thresh)
%                 T{j}(k,l) = all(E < thresh(l));
%             end
%             e{k,i} = E;
%         end
%         err{j,i} = cell2mat(e);
%     end
%     T = cell2mat(T);
%     TT(i,:) = sum(T,1)/size(T,1);
%     % Get figures
%     figure;
%     surf(H);
%     title(config{i,1});
%     xlabel('Tracking steps');
%     ylabel('Distance from the truth (m)');
%     zlabel('Count');
%     set(gca,'Xdir','reverse');
%     set(gca,'Ydir','reverse');
%     set(gca,'YTick',1:length(edges));
%     set(gca,'YTickLabel',edges);
% end
% figure;
% plot(TT');
% title(sprintf('Percentage of successful tracks (~%d steps)',Nduration));
% xlabel('Max distance from the truth (m)');
% set(gca,'XTick',1:length(thresh));
% set(gca,'XTickLabel',thresh);
% legend(strcat(strcat(config(:,1),'+'),config(:,2)),'Location','SouthEast');
% 
% fprintf('Average error\n');
% disp(mean(cell2mat(err),1));
% 
% % Main11 visualization
% files = {...
% 'main11_zara01_f1.mat','main11_zara02_f1.mat','main11_students03_f1.mat',...
% 'main11_zara01_f2.mat','main11_zara02_f2.mat','main11_students03_f2.mat'...
% };
% 
% for f = files
%     load(f{:});
%     aviobj = avifile(['behavior_' D(ts).label '_f' num2str(fold) '.avi'],'fps',2.5);
%     for simid = 1:size(Sims,1)
% %         simVisualize(Sims(simid,:),Obs,Res,methods,D(Sims(simid,2)).video,D(Sims(simid,2)).H);
%         simVisualize(Sims(simid,:),Obs,Res,config(:,1)',D(Sims(simid,2)).video,D(Sims(simid,2)).H);
%         aviobj = addframe(aviobj,gca);
% %         pause;
%     end
%     aviobj = close(aviobj);
% end
% 
% % Main11 error report
% files = {...
% 'main11_zara01_f1.mat','main11_zara02_f1.mat','main11_students03_f1.mat';...
% 'main11_zara01_f2.mat','main11_zara02_f2.mat','main11_students03_f2.mat';...
% };
% 
% for j = 1:size(files,2)
%     % Merge folds
%     f = files(:,j);
%     load(f{1}); Err1 = Err;
%     load(f{2});
%     for i = 1:length(Err)
%         Err{i} = [Err1{i};Err{i}];
%     end
%     %% Report
%     fprintf('\nResults\n');
%     fprintf('||Method  ');
%     for i = 1:size(config,1)
%         fprintf('||%s',config{i,1});
%     end
%     fprintf('||\n');
%     E = zeros(1,size(config,1));
%     fprintf('||Error(m)');
%     for i = 1:size(config,1)
%         E(i) = mean(Err{i});
%         fprintf('||% f',E(i));
%     end
%     fprintf('||\n');
% end
% 
% % Main11 error t-test for each time step, 2-fold cross validation
% clear;
% files = {...
% 'main11_zara01_f1.mat','main11_zara02_f1.mat','main11_students03_f1.mat',...
% 'main11_zara01_f2.mat','main11_zara02_f2.mat','main11_students03_f2.mat';...
% };
% 
% for file_id = 1:length(files)
%     fprintf('%s\n',files{file_id});
%     % Retrieve data
%     load(files{file_id});
%     
%     % Make index structure 'Steps' for time step of each result record
%     Steps = cell(size(config,1),1);
%     for method_id = 1:size(config,1)
%         Steps{method_id} = zeros(size(Res{method_id},1),1);
%     end
%     for sim_id = 1:size(Sims,1) % for each simulation
%         for method_id = 1:size(config,1) % for each method
%             % Retrieve result records
%             ind = Res{method_id}(:,1) == Sims(sim_id,1);
%             if ~any(ind), continue; end
%             Steps{method_id}(ind) = 1:nnz(ind);
%         end
%     end
%     
%     % Aggregate error statistics for each (method, timestep)
%     % Create cell array to store {method,step} => distance samples
%     Nsteps = max(Sims(:,5));
%     D = cell(size(config,1),Nsteps);
%     for method_id = 1:size(config,1)
%         for step_id = 1:Nsteps
%             % Retrieve data records
%             x = Res{method_id}(Steps{method_id}==step_id,:);
%             d = zeros(size(x,1),1);
%             for i = 1:size(x,1)
%                 % Retrieve the corresponding ground truth
%                 y = Obs(Obs(:,1)==x(i,1),:);
%                 y = y(y(:,2)==x(i,2)&y(:,3)==x(i,3),:);
%                 assert(size(y,1)==1);
%                 % Compute distance
%                 d(i) = sqrt((x(i,4)-y(4))^2 + (x(i,5)-y(5))^2);
%             end
%             % Save data samples
%             D{method_id,step_id} = d;
%         end
%     end
%     
%     % Compute t-value for paired methods
%     % This will produce a table (method1,method2,test,p-value,t-value)
%     [i1,i2,i3] = ndgrid(1:size(D,2),1:size(config,1),1:size(config,1));
%     Table = [i3(:) i2(:) i1(:) zeros(numel(i1),3)];
%     ATable = [];
%     for i = 1:size(config,1)
%         for j = i+1:size(config,1)
%             for step_id = 1:size(D,2) % for each step
%                 [h,p,ci,stats] = ttest2(D{i,step_id},D{j,step_id},0.01,[],'unequal');
%                 Table(Table(:,1)==i&Table(:,2)==j&Table(:,3)==step_id,4:6)...
%                     = [h p stats.tstat];
%             end
%             
%             % Aggregation over steps
%             [h,p,ci,stats] = ttest2(cat(1,D{i,:}),cat(1,D{j,:}),0.01,[],'unequal');
%             ATable = [ATable; i j h p stats.tstat];
%         end
%     end
%     Table(Table(:,6)==0,:) = [];
%     
%     % Save
%     Result(file_id) = struct('All',ATable,'PerStep',Table,'Data',{D});
% end
% 
% save main11_stats.mat
% 
% % Print result
% load main11_stats.mat
% fid = fopen('main11_stats.txt','w');
% for file_id = 1:length(files)
%     % Index
%     fprintf(fid,'%s\n',files{file_id});
%     % Aggregated
%     fprintf(fid,' - All\n');
%     fprintf(fid,'    Method1    Method2          h    p-value    t-value\n');
%     for i = 1:size(Result(file_id).All,1)
%         for j = 1:size(Result(file_id).All,2)
%             fprintf(fid,'   % 8.4f',Result(file_id).All(i,j));
%         end
%         fprintf(fid,'\n');
%     end
%     % PerStep
%     fprintf(fid,' - PerStep\n');
%     fprintf(fid,'    Method1    Method2       Step          h    p-value    t-value\n');
%     for i = 1:size(Result(file_id).PerStep,1)
%         for j = 1:size(Result(file_id).PerStep,2)
%             fprintf(fid,'   % 8.4f',Result(file_id).PerStep(i,j));
%         end
%         fprintf(fid,'\n');
%     end
% end
% fclose(fid);
% 
% Another way to see the result
% load main11_stats.mat
% A = zeros(3,12,6);
% B = zeros(3,12,6);
% for file_id = 1:length(Result)
%     A(:,:,file_id) = cellfun(@(y) mean(y),Result(file_id).Data);
%     B(:,:,file_id) = cellfun(@(y) std(y),Result(file_id).Data);
% end
% for i = 1:12
%     fprintf('\\hline\\multirow{3}{*}{N=%d}\n',i);
%     fprintf('& LIN');
%     fprintf('& % 7.4f & % 7.4f \\\\\n',mean(A(1,i,:)),mean(B(1,i,:)));
%     
%     fprintf('& LTA');
%     fprintf('& % 7.4f & % 7.4f \\\\\n',mean(A(2,i,:)),mean(B(2,i,:)));
%     
%     fprintf('& ATTR');
%     fprintf('& % 7.4f & % 7.4f \\\\\n',mean(A(3,i,:)),mean(B(3,i,:)));
% end
% for i = 1:12
%     fprintf('\\hline\\multirow{3}{*}{N=%d}\n',i);
%     
%     [h,p,ci,stats] = ttest2(A(1,i,:),A(2,i,:),[],[],'unequal');
%     fprintf(' & ');
%     if h==0 && stats.tstat<0, fprintf('{\\bf LIN}');
%     else fprintf('LIN'); end
%     fprintf('&  ');
%     if h==0 && stats.tstat>0, fprintf('{\\bf LTA}');
%     else fprintf('LTA'); end
%     fprintf('& % 7.4f & % 7.4f \\\\\n',p,stats.tstat);
%     
%     [h,p,ci,stats] = ttest2(A(1,i,:),A(3,i,:),[],[],'unequal');
%     fprintf(' & ');
%     if h==0 && stats.tstat<0, fprintf('{\\bf LIN}');
%     else fprintf('LIN'); end
%     fprintf('&  ');
%     if h==0 && stats.tstat>0, fprintf('{\\bf ATTR}');
%     else fprintf('ATTR'); end
%     fprintf('& % 7.4f & % 7.4f \\\\\n',p,stats.tstat);
%     
%     [h,p,ci,stats] = ttest2(A(2,i,:),A(3,i,:),[],[],'unequal');
%     fprintf(' & ');
%     if h==0 && stats.tstat<0, fprintf('{\\bf LTA}');
%     else fprintf('LTA'); end
%     fprintf('&  ');
%     if h==0 && stats.tstat>0, fprintf('{\\bf ATTR}');
%     else fprintf('ATTR'); end
%     fprintf('& % 7.4f & % 7.4f \\\\\n',p,stats.tstat);
% end
% 
% % Check the SVM performance
% testGrp('Duration',1);
% testGrp('Duration',2);
% testGrp('Duration',3);
% testGrp('Duration',4);
% testGrp('Duration',5);
% 
% % Debug annotation in total
% % clear;
% % [D,T,Obj] = importData(); save dataset;
% load dataset;
% for did = 1:length(D)
%     c = lines(size(D(did).destinations,1));
%     plot(D(did).obstacles(:,2),D(did).obstacles(:,1),'k+');
%     axis ij;
%     hold on;
%     for i = 1:size(c,1)
%         % Get info
%         persons = D(did).persons(D(did).persons(:,2)==i,1);
%         ind = arrayfun(@(x) any(persons==x),D(did).observations(:,2));
%         P = D(did).observations(ind,3:4);
%         % Display
%         plot(P(:,2),P(:,1),'+','Color',c(i,:));
%         plot(D(did).destinations(i,2),D(did).destinations(i,1),'*',...
%             'Color',c(i,:));
%     end
%     hold off;
%     pause;
% end
% 
% % Export annotated video
% % [D,T,Obj] = importData();
% % save dataset;
% load dataset;
% for did = 1:length(D)
%     aviobj = avifile(['seq_' D(did).label '_annotated.avi'], 'fps',2.5);
%     Ps = [D(did).obstacles ones(size(D(did).obstacles,1),1)]/D(did).H';
%     Ps = [Ps(:,1)./Ps(:,3) Ps(:,2)./Ps(:,3)];
%     for t = unique(D(did).observations(:,1))'
%         % Compute positions
%         X = D(did).observations(D(did).observations(:,1)==t,:);
%         P = [X(:,3:4) ones(size(X,1),1)]/D(did).H';
%         P = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
%         % Get group edges
%         g = arrayfun(@(x)D(did).persons(D(did).persons(:,1)==x,4),X(:,2));
%         E = false(size(X,1));
%         for i = unique(g)', E(g==i,g==i) = true; end
%         for i = size(g,1),  E(i,i+1:end) = false; end
%         [I,J] = find(E);
%         % Plot
%         imshow(read(D(did).video,t));
%         hold on;
%         plot(Ps(:,2),Ps(:,1),'w+');
%         for i = 1:size(I,1)
%             plot([P(I(i),2);P(J(i),2)],[P(I(i),1);P(J(i),1)],...
%                 '-o','Color','r','LineWidth',1)
%         end
%         for i = 1:size(X,1)
%             text(P(i,2),P(i,1)+10,cellstr(num2str(X(i,2))),...
%                 'Color','r','FontSize',12,'FontWeight','bold');
%         end
%         text(5,10,sprintf('Frame %d',t),'Color','w',...
%             'FontSize',12,'FontWeight','bold');
%         hold off;
%         aviobj = addframe(aviobj,gca);
%     end
%     aviobj = close(aviobj);
%     close all;
% end
% 
% % FFMPEG
% for did = 1:length(D)
%     system(['/opt/local/bin/ffmpeg -y -i '...
%             'seq_' D(did).label '_annotated.avi '...
%             '-b 1.0M -vcodec libx264 -vpre hq '...
%             'seq_' D(did).label '_annotated.mp4']);
% end
% 
% % UCY static object annotation
% vobj = VideoReader('ucy_crowd/data_students03/video.avi');
% % vobj = mmreader('ucy_crowd/data_zara01/video.avi');
% imshow(read(vobj,1));
% [I,X,Y] = roipoly();
% P = [X,Y];
% save ucy_crowd/data_students03/static.txt P -ascii
% % save ucy_crowd/data_zara01/static.txt P -ascii
% clear vobj
% 
% % UCY homography calculation
% % BMW 3 series
% % Width: 70.2 in = 1.78308 meters
% % Length: 181.9 in = 4.62026 meters
% 
% % data_zara
% % Measurement in world
% p = [0.00000 0.00000 1;...
%      4.62026 0.00000 1;...
%      4.62026 1.78308 1;...
%      0.00000 1.78308 1];
% % Measurement in image coords
% % imshow(read(mmreader('ucy_crowd/data_zara01/video.avi'),1));
% % [I,X,Y] = roipoly();
% % P = [Y(1:end-1) X(1:end-1) ones(size(X,1)-1,1)];
% P = [302.0000 476.0000 1.0000;...
%      124.0000 472.0000 1.0000;...
%      124.0000 549.0000 1.0000;...
%      302.0000 563.0000 1.0000];
% % Homography
% H = (P\p)';
% disp(P*H');
% save ucy_crowd/data_zara01/H.txt H -ascii
% 
% % Measurement in world
% p = [0.00000 0.00000 1;...
%      3.00000 0.00000 1;...
%      3.00000 2.95000 1;...
%      0.00000 2.95000 1];
% % Measurement in image coords
% % imshow(read(mmreader('ucy_crowd/data_students03/video.avi'),1));
% % [I,X,Y] = roipoly();
% % P = [X(1:end-1) Y(1:end-1) ones(size(X,1)-1,1)];
% P = [384.0000 318.0000 1.0000;
%      254.0000 322.0000 1.0000;
%      257.0000 465.0000 1.0000;
%      388.0000 476.0000 1.0000];
% % Homography
% H = (P\p)';
% disp(P*H');
% save ucy_crowd/data_students03/H.txt H -ascii
% 
% % UCY destination annotation
% p = [...
%     -15  1;...
%     -12  12;...
%     16 6;...
%     16 -2;...
%     5 16;...
%     1 -12;...
%     ];
% save ucy_crowd/data_students03/destinations.txt p -ascii
% 
% seq = ucyLoad();
% s = seq.students03;
% persons = unique(s.obsmat(:,3))';
% 
% % Assign destinations
% dest = s.destinations;
% cmap = lines(size(dest,1));
% IDX = ones(size(dest,1),1);
% for i = 1:length(persons)
%     P = s.obsmat(s.obsmat(:,3)==persons(i),4:5);
%     d = (P(end,1)-dest(:,1)).^2+(P(end,2)-dest(:,2)).^2;
%     IDX(i) = find(d==min(d));
% end
% 
% % Plot
% plot(s.destinations(:,1),s.destinations(:,2),'k*');
% hold on;
% for i = 1:length(persons)
%     P = s.obsmat(s.obsmat(:,3)==persons(i),4:5);
%     plot(P(:,1),P(:,2),'Color',cmap(IDX(i),:));
%     plot(P(end,1),P(end,2),'*','Color',cmap(IDX(i),:));
% end
% hold off;
% 
% % UCY annotation debug
% seq = ucyLoad();
% datasets = fieldnames(seq);
% for did = 3%1:length(datasets)
%     aviobj = avifile(['seq_' datasets{did} '.avi'],'fps',2.5);
%     s = seq.(datasets{did});
%     for t = unique(s.obsmat(:,2))'
%         % compute image coords
%         x = s.obsmat(s.obsmat(:,2)==t,:);
%         P = x(:,4:5);
%         P = [P ones(size(P,1),1)]/s.H';
%         P = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
%         % render
%         imshow(read(s.video,t));
%         hold on;
%         plot(P(:,1),P(:,2),'ro');
%         text(5,10,sprintf('Frame %d',t),'Color','w');
%         text(P(:,1)+5,P(:,2)+10,cellstr(num2str(x(:,3))),'Color','r');
%         hold off;
%         drawnow;
%         aviobj = addframe(aviobj,gca); 
%     end
%     aviobj = close(aviobj);
%     system(['/opt/local/bin/ffmpeg -i seq_' datasets{did} '.avi '...
%             '-b 1024k -vcodec libx264 -vpre medium seq_' datasets{did} '.mp4']);
% end
% 
% % Manual annotation of static objects
% load ewap_dataset;
% seqShow(seq.eth,seq.eth.groups,seq.eth.obsmat(1));
% [B,X,Y] = roipoly;
% if isempty(X), keyboard; end
% P = [X Y];
% save ewap_dataset/seq_eth/static.txt P -ascii;
% seq = ewapLoad('ewap_dataset');
% save ewap_dataset seq;
% 
% % Cluster end of trajectories
% seq = ewapLoad('ewap_dataset');
% datasets = fieldnames(seq);
% [D,Obj] = seq2ewap(seq);
% edges = linspace(-pi,pi,9);
% for did = 1%unique(D(:,1)')
%     dest = seq.(datasets{did}).destinations;
%     cmap = lines(size(dest,1));
%     persons = unique(D(D(:,1)==did&D(:,13)==1,3)');
%     
%     % Assign destinations
%     IDX = zeros(length(persons),1);
%     for i = 1:length(persons)
%         Pdest = D(find(D(:,1)==did&D(:,3)==persons(i),1),10:11);
%         IDX(i) = find(all(dest==repmat(Pdest,[size(dest,1),1]),2));
%     end
%     % Plot
%     hold on;
%     for i = 1:length(persons)
% %         if IDX(i)~=3&&IDX(i)~=4, continue; end
%         T = D(D(:,1)==did&D(:,3)==persons(i),4:5);
%         plot(T(:,1),T(:,2),'Color',cmap(IDX(i),:));
%         plot(T(end,1),T(end,2),'+','Color',cmap(IDX(i),:));
%     end
%     for i = 1:size(dest,1)
%         plot(dest(i,1),dest(i,2),'*','Color',cmap(i,:));
%     end
%     hold off;
% end
% 
% % Query maximum pairwise distance of grouped people
% seq = ewapLoad('ewap_dataset/');
% [D,Others] = seq2ewap(seq);
% datasets = fieldnames(seq);
% for d_id = 1:length(datasets)
%     % Query data from dataset d_id
%     T = D(D(:,1)==d_id,:);
%     O = Others(D(:,1)==d_id);
%     % Compute pairwise distance
%     d = zeros(size(T,1),1);
%     for i = 1:size(T,1)
%         Pi = T(i,[4 5]);
%         Pj = O{i}(logical(O{i}(:,6)),[2 3]);
%         if isempty(Pj), Pj = Pi; end
%         d(i) = max(sqrt((Pi(1) - Pj(:,1)).^2 + (Pi(2) - Pj(:,2)).^2));
%     end
%     % Find maximum distance and show info
%     [dmax,id] = max(d);
%     fprintf('Dataset %d: ', d_id);
%     fprintf('Distance=%f[m]\n', dmax);
%     disp(T(id,[3 4 5 6 7]));
%     disp(O{id}(logical(O{id}(:,6)),:));
%     disp(find(unique(seq.(datasets{d_id}).obsmat(:,1))==T(id,2)));
% end
% 
% % Group estimation
% % Estimate groups
% % load ewap_mod_results
% % [G,M,A] = ewapError2(D,Others,...
% %     [0.1301    2.0879    2.3272    2.0731    1.4643    0.8032    0.9999]);
% % save;
% load;
% % Convert Format
% groups = [cell2mat(G{1});...
%           cell2mat(arrayfun(@(i) i*ones(1,length(G{1}{i})),...
%                    1:length(G{1}),'UniformOutput',false))]';
% % add singleton missing from the dataset
% orphans = setdiff(unique(seq.eth.obsmat(:,2)),unique(groups(:,1)));
% groups = [groups; orphans (1:length(orphans))'+size(groups,1)];
% 
% % Render
% seq = ewapLoad('ewap_dataset/');
% seqShow(seq.eth,groups);
% 
% %% Debug error function
% load ewap_dataset;
% [D,Obj] = seq2ewap(seq);
% nfolds = 3;
% % params = [0.143719 2.051546 2.719708 1.666738 1.312681 0.588800];
% params = [0.130081 2.087902 2.327072 2.0732 1.461249...
%           0.500000 1.127248 0.500000];
% Dind = unique(D(:,[1 3]),'rows'); % Unique pairs of (dataset,person_id)
% for expId = 1%:nfolds
%     % Prepare index of testing samples
%     Train = Dind(mod(1:size(Dind,1),nfolds)==expId-1,:);
%     Test  = Dind(mod(1:size(Dind,1),nfolds)~=expId-1,:);
%     % Estimate groups
%     fprintf('Fold: %d\n',expId);
%     tic;
%     E = attractionError(D,Obj,params,'Index',Train);
%     disp(mean(E));
%     toc;
% end
% load ewap_dataset;
% [D,Obj] = seq2ewap(seq);
% nfolds = 5;
% params = [0.143719 2.051546 2.719708 1.666738 1.312681 0.588800];
% for expId = 1%:nfolds
%     % Prepare index of testing samples
% %     ind = mod(1:size(D,1),nfolds) == expId-1;
%     ind = ceil(nfolds*(1:size(D,1))/(size(D,1))) == expId;
%     % Estimate groups
%     fprintf('Fold: %d\n',expId);
%     tic;
% %     E = ewapError3(D(ind,:),Obj,params);
%     E = linError(D(ind,:));
%     toc;
% end
