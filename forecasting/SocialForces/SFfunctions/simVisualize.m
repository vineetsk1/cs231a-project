function [ ] = simVisualize( Sims, X, Xhat, labels, vid, H )
%SIMVISUALIZE Summary of this function goes here
%
%   Sims(simulation,dataset,person,start,duration)
%   X(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Xhat{i}(simulation,time,person,px,py,vx,vy,dest,speed)

% Drop past items
X = X(X(:,1)==Sims(1)&X(:,2)>=Sims(4)&X(:,3)==Sims(3),:);
for i = 1:length(Xhat)
    Xhat{i} = Xhat{i}(Xhat{i}(:,1)==Sims(1)&...
                      Xhat{i}(:,2)>Sims(4)&...
                      Xhat{i}(:,3)==Sims(3),:);
end

% Load image
img = read(vid,Sims(4));

% Convert coorinates
P = [X(:,4:5) ones(size(X,1),1)] / H';
X(:,4:5) = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
for i = 1:length(Xhat)
    P = [Xhat{i}(:,4:5) ones(size(Xhat{i},1),1)] / H';
    Xhat{i}(:,4:5) = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
end

% Plot
P = zeros(size(X,1),1+length(Xhat),2);
P(:,1,1) = X(:,5); P(:,1,2) = X(:,4);
for i = 1:length(Xhat)
     P(:,i+1,1) = [X(1,5);Xhat{i}(:,5)];
     P(:,i+1,2) = [X(1,4);Xhat{i}(:,4)];
end

imshow(img);
hold on;
l = {'-*','-x','-o','-+'};
c = jet(1+length(Xhat));
h = zeros(1+length(Xhat),1);
for i = 1:length(h)
    h(i) = plot(P(:,i,1),P(:,i,2),l{mod(i-1,length(l))+1},'Color',c(i,:),...
        'LineWidth',2,'MarkerSize',10);
end
legend(h,['TRUTH  ' labels],'FontSize',12,'FontWeight','bold');
text(5,10,...
    sprintf('Sequence: %d, Frame: %d, Person: %d',Sims(1),Sims(4),Sims(3)),...
    'Color','w','FontSize',12,'FontWeight','bold');
hold off;

end

