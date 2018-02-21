clear all
close all
% 
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
% 
% pause



clear all
close all

Is = load('C:\Users\Alexandre\Dropbox\Stages\Stanford\alex-jackrabbot\toy\1\static\sidewalk.mat');
Is = Is.arr;

% destination estimation
nd=1;
for i = [1 size(Is,1)]
    D{nd} = mean([abs(size(Is,1)-find(Is(i,:)==1))',i*ones(length(find(Is(i,:)==1)),1)]); %#ok<*SAGROW>
    if ~isempty(D{nd})
        nd = nd+1;
    end
    D{nd} = mean([i*ones(length(find(Is(:,i)==1)),1) abs(size(Is,1)-find(Is(:,i)==1))]);
    if ~isempty(D{nd})
        nd = nd+1;
    end
end

% Convert the image to a speed map

SpeedImage=Is*1000+0.001;
dest = (1:nd-1);
Sp = []; Spc={}; k=0; Path = [];

for n = dest
    d = dest;
    figure
    hold on
    % Set the source to end of the map
    SourcePoint=[D{n}(1);D{n}(2)];
    % Calculate the distance map (distance to source)
    DistanceMap= msfm(SpeedImage, SourcePoint);
    d(d==n)=[];
    imshow(DistanceMap,[0 30000])
    hold on
    plot(SourcePoint(2),SourcePoint(1),'*b')
    
    for i=d
        k=k+1;
        StartPoint=[D{i}(1);D{i}(2)];
        ShortestLine=shortestpath(DistanceMap,StartPoint,SourcePoint);
        hold on, plot(ShortestLine(:,2),ShortestLine(:,1),'r');
        Spc{k}=[n*ones(length(ShortestLine),1) i*ones(length(ShortestLine),1)...
            ShortestLine];
        Sp = [Sp; Spc{k}]; %#ok<AGROW>
        Path = [Path; k n i ]; %#ok<AGROW>
    end
    
end

L =(1:length(Spc)); Int=[]; intpath={}; IntLabel=[]; j=1;

for k=L
    l = L;
    l(l==k)=[];
    plot(Spc{k}(:,4),Spc{k}(:,3),'g');
    
    for i=l
        [a,b] = intersections(Spc{k}(150:end-100,4),Spc{k}(150:end-100,3),...
            Spc{i}(150:end-100,4),Spc{i}(150:end-100,3),2);
        if ~isempty(a)
            Int = [Int ;a b]; %#ok<AGROW>
        end
        IntLabel = [IntLabel; j k i]; %#ok<AGROW> %(label intersection, path 1, path 2 )
        intpath{j}=[j*ones(length(a),1) a b];
        j=j+1;
    end
    
    Dff{k} = Spc{k}(abs(diff(Spc{k}(:,3),2))>quantile(diff(Spc{k}(:,3),2),.9),:);
    e = evalclusters(Dff{k}(:,3:4),'kmeans','CalinskiHarabasz','KList',[1:3]);
    [idx,Dff{k}]=kmeans(Dff{k}(:,3:4),e.OptimalK,'start','cluster');
    plot(Dff{k}(:,2),Dff{k}(:,1),'y*');
end

%plot(Dff{k}(:,4),Dff{k}(:,3),'r*')

LocalGoal = Dff;



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