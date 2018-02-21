function [ L ] = salvage( S, R, config, varargin )
%SALVAGE Summary of this function goes here
%   Detailed explanation goes here

maxDist = 1.0; % meter
stepRange = [1 inf];
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'MaxDist'), maxDist = varargin{i+1}; end
    if strcmp(varargin{i},'StepRange'), stepRange = varargin{i+1}; end
end

% Scan the result and associate tracker label by nearest neighbor
L = cell(length(S),size(config,1));
for i = 1:length(S)
    for j = 1:size(config,1)
        % Check if track is correct
        frames = unique(R{i,j}.obsv(:,1))';
        l = cell(length(frames),1);
        for t = 1:length(frames)
            % Predicted/True records at time t: (t,pid,px,py,vx,vy,dest,u)
            pre = R{i,j}.obsv(R{i,j}.obsv(:,1)==frames(t),:);
            persons = unique(pre(:,2));
            
            % Filter by #tracking step
            step = t + 1 - arrayfun(@(x) find(frames==...
                S(i).trks(S(i).trks(:,1)==x,2)),persons);
            persons = persons(stepRange(1) <= step & step <= stepRange(2));
            pre = pre(arrayfun(@(k) any(k==persons),pre(:,2)),:);
            
            % Get ground truth
            tru = S(i).obsv(S(i).obsv(:,1)==frames(t),:);
            tru = tru(arrayfun(@(k) any(k==persons),tru(:,2)),:);
            % Compute distance and assign label
            l{t} = zeros(size(pre,1),4);
            for k = 1:size(pre,1)
                d = sqrt((tru(:,3)-pre(k,3)).^2 + (tru(:,4)-pre(k,4)).^2);
                dtru = d(tru(:,2)==pre(k,2)); % Keep the true distance
                d(tru(:,5:6)*pre(k,5:6)'<0) = inf; % ignore opposite dir
                [dmin,ind] = min(d);
                if dmin<=maxDist
                    % Get person id of the closest if not lost
                    % (frame, truth, nearest)
                    l{t}(k,:) = [frames(t) pre(k,2) tru(ind,2) dtru];
                else
                    l{t}(k,:) = [frames(t) pre(k,2) 0 dtru];
                end
            end
        end
        L{i,j} = cell2mat(l);
    end
end

end

