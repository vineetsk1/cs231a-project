function [ ] = seqShow( seq, groups, frame )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

C = lines(16);
Time = unique(seq.obsmat(:,1))';

if ~exist('frame','var'), frame = Time; end
for t = frame
    % Query records at time t
    ind = seq.obsmat(:,1)==t;
    D = seq.obsmat(ind,:);
    
    P = [D(:,3) D(:,5) ones(nnz(ind),1)] / seq.H';
    P = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
    c = mod(groups(arrayfun(@(j) find(groups(:,1)==j),D(:,2)),2),16)+1;
    % Plot
    imshow(read(seq.video,t));
    hold on;
    scatter(P(:,2),P(:,1),100,C(c,:));
    plot(seq.static(:,1),seq.static(:,2),'w+');
    drawnow;
    for i = 1:length(c)
        text(P(i,2),P(i,1)+10,cellstr(num2str(D(i,2))),'Color',C(c(i),:));
    end
    hold off;
end

end

