function [D,Path,LocalGoal,Obst] = mapanalysis(map,im,destinations)

if nargin==0
    map='/scr/alexz/from_local/toy_dataset/0/static/sidewalk.mat';
end
if nargin<2
    im = 0;
end


% % Load a maze image
% I1=im2double(imread('images/maze.gif'));
%
% % Convert the image to a speed map
% SpeedImage=I1*1000+0.001;
%
% % Set the source to end of the maze
% SourcePoint=[800;803];
% imshow(I1,[0 0.0039]);
% hold on
% plot(SourcePoint(1),SourcePoint(2),'r*')
% pause
% close all
% % Calculate the distance map (distance to source)
% DistanceMap= msfm(SpeedImage, SourcePoint);
% % Show the distance map
% figure, imshow(DistanceMap,[0 3400])
% pause
% % Trace shortestline from StartPoint to SourcePoint
% StartPoint=[9;14];
% ShortestLine=shortestpath(DistanceMap,StartPoint,SourcePoint);
% % Plot the shortest route
% hold on, plot(ShortestLine(:,2),ShortestLine(:,1),'r')

%
% pause
%
%
%
% clear all
% close all

Is = load(map);
Is = Is.arr;
Obst = imcontour(Is)';close all
%Obst(:,2)=abs(Obst(:,2)-max(Obst(:,2)));
% destination estimation
if nargin<3
    nd=1;
    for i = [1 size(Is,1)]
        D(nd,:) = mean([abs(find(Is(i,:)==1))',i*ones(length(find(Is(i,:)==1)),1)]); %#ok<*SAGROW>
        if ~isempty(D(nd,:))
            nd = nd+1;
        end
        D(nd,:) = mean([i*ones(length(find(Is(:,i)==1)),1) abs(find(Is(:,i)==1))]);
        if ~isempty(D(nd,:))
            nd = nd+1;
        end
    end
else
    dest = importdata(destinations);
    nd=size(dest,1);
    D=dest(:,[2 3]);
    nd = nd+1;
end


% Convert the image to a speed map

SpeedImage=Is*1000+0.001;
dest = (1:nd-1);
Sp = []; Spc={}; k=0; Path = [];

for n = dest
    d = dest;
    
    % Set the source to end of the map
    SourcePoint=[D(n,2);D(n,1)];
    % Calculate the distance map (distance to source)
    DistanceMap= msfm(SpeedImage, SourcePoint);
    d(d==n)=[];
    if im ==1
        figure
        imshow(DistanceMap,[0 30000])
        hold on
        plot(SourcePoint(2),SourcePoint(1),'*b');
    end
    
    for i=d
        k=k+1;
        StartPoint=[D(i,2);D(i,1)];
        ShortestLine=shortestpath(DistanceMap,StartPoint,SourcePoint);
        
        if im==1
            plot(ShortestLine(:,2),ShortestLine(:,1),'r');
        end
        
        
        Spc{k}=[n*ones(length(ShortestLine),1) i*ones(length(ShortestLine),1)...
            ShortestLine];
        Sp = [Sp; Spc{k}];
        Path = [Path; k n i ];
    end
    
end

L =(1:length(Spc));
Int=[]; intpath={}; IntLabel=[]; j=1;

k=4;c=[];ok=0;
while ok==0
    c = corner(Is,k);
    c11 = sum(c(:,1)==0);
    c12 = sum(c(:,1)==320);
    c21 = sum(c(:,2)==0);
    c22 = sum(c(:,2)==320);
    if c11+c12+c21+c22>k-4
        ok=0;
    else
        c(c(:,1)==0,:)=[];
        c(c(:,1)==320,:)=[];
        c(c(:,2)==0,:)=[];
        c(c(:,2)==320,:)=[];
        ok=1;
    end
    k=k+1;
end

for k=L
    l = L;
    l(l==k)=[];
    if im==1
        plot(Spc{k}(:,4),Spc{k}(:,3),'g');
    end
    %     for i=l
    %         [a,b] = intersections(Spc{k}(150:end-100,4),Spc{k}(150:end-100,3),...
    %             Spc{i}(150:end-100,4),Spc{i}(150:end-100,3),2);
    %         if ~isempty(a)
    %             Int = [Int ;a b];
    %         end
    %         IntLabel = [IntLabel; j k i];%(label intersection, path 1, path 2 )
    %         intpath{j}=[j*ones(length(a),1) a b];
    %         j=j+1;
    %     endabs(Spc{k}

    d = 1000;
    for j=1:length(c)
        [nd,md] = min(sum(abs(Spc{k}(:,[4 3])-repmat(c(j,:),length(Spc{k}),1)),2));
        if nd<d
            mp = j;d=nd;
        end
    end
       
    Dff{k} = c(mp,:);
%     Dff{k} = Spc{k}(abs(diff(Spc{k}(:,4),2))+abs(diff(Spc{k}(:,3),2))...
%         >quantile(abs(diff(Spc{k}(:,4),2))+abs(diff(Spc{k}(:,3),2)),.99),:);
%     %e = evalclusters(Dff{k}(:,3:4),'kmeans','CalinskiHarabasz','KList',1:4);
%     
%     %Dff{k} =  Spc{k}(max(diff(Spc{k}(:,3),2)),:)
%     [idx,Dff{k}]=kmeans(Dff{k}(:,3:4),2,'start','sample'); %#ok<*ASGLU,*AGROW>
%     
%     
%     Dff{k}=Dff{k}(:,[2 1]);
%     [~,ord]=sort(sum(abs(Dff{k}-repmat(D(Path(k,2),:),size(Dff{k},1),1)),2));
%     Dff{k} = Dff{k}(ord,:);
    if im==1
        plot(Dff{k}(:,1),Dff{k}(:,2),'y*');
    end
end
%plot(Dff{k}(:,4),Dff{k}(:,3),'r*')
%Dff(:,[1 2])=Dff(:,[2 1]);
LocalGoal = Dff;
%D = D(:,[2 1]);
for k=1:length(LocalGoal)
    [~,v]=sort(sum(abs(LocalGoal{k}-repmat(D(Path(k,2),:),size(LocalGoal{k},1),1)),2));
    LocalGoal{k}=LocalGoal{k}(v,:);
end





%     for k=1:length(D{n});
%         % Set the source to end of the map
%         if length(D{n})==2
%             SourcePoint=[D{n}(2);D{n}(1)];
%         else
%
%             SourcePoint=[D{n}(k,2);D{n}(k,1)];
%         end
%         % Calculate the distance map (distance to source)
%         DistanceMap= msfm(SpeedImage, SourcePoint);
%         d(d==n)=[];
%         hold on
%         imshow(DistanceMap,[0 50000])
%         plot(SourcePoint(1),SourcePoint(2),'*b')
%
%         for i=d
%             for j=1:length(D{i})
%                 StartPoint=[D{i}(2);D{i}(1)];
%                 ShortestLine=shortestpath(DistanceMap,StartPoint,SourcePoint);
%                 hold on, plot(ShortestLine(:,2),ShortestLine(:,1),'r');
%             end
%             %             for j=1:length(D{i})
%             %                 StartPoint=[D{i}(j,2);D{i}(j,1)];
%             %                 ShortestLine=shortestpath(DistanceMap,StartPoint,SourcePoint);
%             %                 hold on, plot(ShortestLine(:,2),ShortestLine(:,1),'r');
%             %             end
%             pause
%         end
%
%     end
%end

end
