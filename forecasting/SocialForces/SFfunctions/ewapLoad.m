function [ seq ] = ewapLoad( input_dir )
%EWAPREAD function to import ewap dataset
%   Detailed explanation goes here

if input_dir(end) ~= '/', input_dir = [input_dir '/']; end

% ETH sequence
seq.eth.destinations = load([input_dir 'seq_eth/destinations.txt']);
seq.eth.obsmat = load([input_dir 'seq_eth/obsmat.txt']);
seq.eth.H = load([input_dir 'seq_eth/H.txt']);
seq.eth.map = imread([input_dir 'seq_eth/map.png']);
seq.eth.groups = pedestrian_group_read(seq.eth.obsmat,...
                                       [input_dir 'seq_eth/groups.txt']);
%if exist('mmreader','class')
    seq.eth.video = VideoReader([input_dir 'seq_eth/video.avi']);
%end
% seq.eth.walkable = ewapObstacle(seq.eth);
seq.eth.static = load([input_dir 'seq_eth/static.txt']);

% HOTEL sequence
seq.hotel.destinations = load([input_dir 'seq_hotel/destinations.txt']);
seq.hotel.obsmat = load([input_dir 'seq_hotel/obsmat.txt']);
seq.hotel.H = load([input_dir 'seq_hotel/H.txt']);
seq.hotel.map = imread([input_dir 'seq_hotel/map.png']);
seq.hotel.groups = pedestrian_group_read(seq.hotel.obsmat,...
                                       [input_dir 'seq_hotel/groups.txt']);
%if exist('mmreader','class')
    seq.hotel.video = VideoReader([input_dir 'seq_hotel/video.avi']);
%end
% seq.hotel.walkable = ewapObstacle(seq.hotel);
seq.hotel.static = load([input_dir 'seq_hotel/static.txt']);

end

%%
function [ groups ] = pedestrian_group_read( obsmat, input_file )
% PEDESTRIAN_GROUP_READ returns groups(person_id, group_id) table

% Load non-empty groups
g = zeros(0,2);
id = 1;
fid = fopen(input_file,'r');
while ~feof(fid)
    l = sscanf(fgetl(fid),'%d');
    if ~isempty(l)
        g = [g;[repmat(id,[length(l) 1]) l]];
        id = id + 1;
    end
end
fclose(fid);

% Add singletons
id = 1;
groups = unique(obsmat(:,2));
groups = [groups zeros(size(groups,1),1)];
for i = 1:size(groups,1)
    if groups(i,2) == 0
        ind = groups(i,1);
        gid = g(g(:,2)==ind,1);
        if ~isempty(gid)
            ind = union(ind,g(arrayfun(@(x) any(x==gid),g(:,1)),2));
        end
        groups(arrayfun(@(x) any(x==ind),groups(:,1)),2) = id;
        id = id + 1;
    end
end

end

%%
function [ map ] = ewapObstacle( seq, varargin )
%EWAPOBSTACLE creates obstacle density map
%   Detailed explanation goes here

% Options
% LineWidth = 3;
% GaussianSize = [11 11];
% GaussianSigma = 2.0;
for i = 1:2:length(varargin)
%     if strcmpi(varargin{i},'LineWidth')
%         LineWidth = varargin{i+1};
%     end
%     if strcmpi(varargin{i},'GaussianSize')
%         GaussianSize = varargin{i+1};
%     end
%     if strcmpi(varargin{i},'GaussianSigma')
%         GaussianSigma = varargin{i+1};
%     end
end

% Allocate memory
if isfield(seq,'video') && isa(seq.video,'mmreader')
    map = false(seq.video.Height,seq.video.Width);
else
    map = false(480,640);
end

persons = unique(seq.obsmat(:,2));
H = seq.H;

% Screen plot
p = cell(1,length(persons));
for i = 1:length(persons)
    p{i} = seq.obsmat(seq.obsmat(:,2) == persons(i),[3 5]);
    p{i} = [p{i} ones(size(p{i},1),1)] / H';
    p{i} = round(p{i}(:,1:2) ./ repmat(p{i}(:,3),[1 2]));
end

% Draw trajectory map
for i = 1:length(persons)
    for j = 1:size(p{i},1)-1
        map = bresenham(map,p{i}(j,[1 2]),p{i}(j+1,[1 2]));
    end
end

% % Smooth the map
% map = imfilter(double(map),...
%        fspecial('gaussian',GaussianSize,GaussianSigma));

% Distance transform
map = bwdist(map);

end

%%
function [ z ] = bresenham( z, p1, p2 )
%BRESENHAM draws line on input logical matrix z
%
% Input:
%  z:  2D logical matrix
%  p1: 1-by-2 vector: line start
%  p2: 1-by-2 vector: line end
%

% Check the slope
steep = abs(p1(1) - p2(1)) > abs(p1(2) - p2(2));
if steep % Swap x and y
    p1 = fliplr(p1);
    p2 = fliplr(p2);
end

% Check the direction
if p1(2) > p2(2) % Swap start and end
    t = p1;
    p1 = p2;
    p2 = t;
end

% Prepare integers
dx = int32(p2(2) - p1(2));
dy = int32(abs(p2(1) - p1(1)));
err = int32(dx * 0.5);
y = p1(1);

% Decide y direction
if p1(1) < p2(1)
    ystep = 1;
else
    ystep = -1;
end

% Draw
for x = p1(2):p2(2)
    % Fill in the pixel
    if steep
        z(x,y) = true;
    else
        z(y,x) = true;
    end
    
    % Update y
    err = err - dy;
    if err < 0
        y = y + ystep;
        err = err + dx;
    end
end

end