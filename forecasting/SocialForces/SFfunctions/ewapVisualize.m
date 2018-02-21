function [ I, Phat ] = ewapVisualize( I, H, D, Others, params, varargin )
%EWAPVISUALIZE draws energy map on top of the input image
%   []
%
% Input:
%   I: Input image
%   H: Homography matrix
%   D: Table of features
%   Others: Cell array of additional features attached to each row of D
%   params: Parameters of energy function
% Output:
%   I: Output image
%   Phat: Predicted position for each decision in the world coordinates

% Options
a = 0.25;          % alpha value of the energy map
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Alpha'), a = varargin{i+1}; end
end

% Colormap
C = round(255*reshape(jet(256),[1 256 3]))+1;

% Get ROI
[R,Ri] = ewapROI(D,H,'ImageSize',size(I));

% Compute energy and Pnext for each position
E = cell(size(R));
if nargout > 1, Phat = zeros(length(R),2); end
for k = 1:length(R) % For each pedestrian
    % Compute energy at each location in ROI
    E{k} = zeros(size(R{k},1),1);
    for i = 1:size(R{k},1) % For each location
        vhat = (2.5*(R{k}(i,:)-D(k,[4 5])) - ...
                D(k,[6 7])*params(6))/(1-params(6));
        E{k}(i) = ewapEnergy(vhat,...
            [D(k,[4 5]);Others{k}(:,[1 2])],...
            [D(k,[6 7]);Others{k}(:,[3 4])],...
            D(k,12),D(k,[10 11]),params);
    end
    % Normalize the scale
    E{k} = round(255*((E{k}-min(E{k})) / (max(E{k})-min(E{k}))))+1;
    E{k}(isnan(E{k})) = 1;  % When only one element exists
    if nargout > 1
    % Pnext
        Phat(k,:) = ewapPredict(...
            [D(k,[4 5]);Others{k}(:,[1 2])],...
            [D(k,[6 7]);Others{k}(:,[3 4])],...
            D(k,12),D(k,[10 11]),params);
    end
end

% Draw
for k = 1:length(Ri)
    for i = 1:size(Ri{k},1)
%         if k > length(Ri) || k > length(E) || i > size(Ri{k},1) || ...
%             i > length(E{k}) || E{k}(i) > size(C,2) ||...
%             Ri{k}(i,1) > size(I,1) || Ri{k}(i,2) > size(I,2)
%             keyboard;
%         end
        I(Ri{k}(i,1),Ri{k}(i,2),:) = ...
            round((1-a) * C(1,E{k}(i),:) +...
                   a * double(I(Ri{k}(i,1),Ri{k}(i,2),:)));
    end
end
% 
% % Plot (optional);
% imshow(I);
% hold on;
% Phat = [Phat ones(length(R),1)] / H';
% plot(Phat(:,2)./Phat(:,3),Phat(:,1)./Phat(:,3),'r+');
% Plin = [D(ind,4:5)+0.4*D(ind,6:7) ones(length(R),1)] / H';
% plot(Plin(:,2)./Plin(:,3),Plin(:,1)./Plin(:,3),'g+');
% Ptruth = [D(ind,8) D(ind,9) ones(length(R),1)] / H';
% plot(Ptruth(:,2)./Ptruth(:,3),Ptruth(:,1)./Ptruth(:,3),'wo');
% hold off;
% legend({'LTA','LIN','Truth'});

end

function [ ROI, ROIpxl ] = ewapROI( D, H, varargin )
%PICKROI specifies ROI of each decision making for visualization
%   Detailed explanation goes here

% Options
format = 'ewap';          % Dataset format
imagesize = [480 640 3];  % [Width Height Channel]
accels = [0.25 1.75];   % min and max of acceleration range
angles = [-85 85] * pi / 180;   % min and max of angle range
for i = 1:2:length(varargin)
    if strcmp(varargin{i},'Format'), format = varargin{i+1}; end
    if strcmp(varargin{i},'ImageSize'), imagesize = varargin{i+1}; end
    if strcmp(varargin{i},'AccelRange'), accels = varargin{i+1}; end
    if strcmp(varargin{i},'AngleRange'), angles = varargin{i+1}; end
end

% Preprocessing to convert velocity to polar representation
if strcmp(format,'ewap')
    [D(:,7),D(:,6)] = cart2pol(D(:,6),D(:,7));
end

axsdir = [-0.5 0.0 0.5 1.0] * pi;
ROI = cell(1,size(D,1));
ROIpxl = cell(1,size(D,1));
for i = 1:size(D,1)
    % Retrieve info
    p0 = D(i,[4 5]);    % position
    rho0 = 0.4*D(i,6);  % magnitude of velocity
    theta0 = D(i,7);    % angle of velocity
    % Corners
    [x,y] = pol2cart(theta0+angles([1;1;2;2])',...
                     rho0 * accels([1;2;1;2])');
    % Outermost points in axis direction
    ind = ([cos(axsdir') sin(axsdir')] * [cos(theta0) sin(theta0)]') > 0;
    [xe,ye] = pol2cart(axsdir(ind)',rho0*accels(2));
    % Compute bounding box of the ROI
    bx = [min([x;xe]+p0(1)) max([x;xe]+p0(1))];
    by = [min([y;ye]+p0(2)) max([y;ye]+p0(2))];
    bbox = [bx(1) by(1);bx(1) by(2);bx(2) by(2);bx(2) by(1)];
    % Apply inverse homography transformation
    bbox = [bbox ones(4,1)] / H';
    bbox = bbox(:,1:2) ./ repmat(bbox(:,3),[1 2]);
    % Recalculate bounding box in pixel coordinate
    bx = round([min(bbox(:,1)) max(bbox(:,1))]);
    by = round([min(bbox(:,2)) max(bbox(:,2))]);
    bx(bx<1) = 1; by(by<1) = 1;
    bx(bx>imagesize(1)) = imagesize(1);
    by(by>imagesize(2)) = imagesize(2);
    [X,Y] = meshgrid(bx(1):bx(2),by(1):by(2));    
    % Apply transformation
    p = [X(:) Y(:) ones(numel(X),1)] * H';
    p = p(:,1:2) ./ repmat(p(:,3),[1 2]);
    % See condition inside the disk
    [theta,rho] = cart2pol(p(:,1)-p0(1),p(:,2)-p0(2));
    ind = rho0*accels(1) <= rho & rho <= rho0*accels(2) &...
        ((angles(1)     +theta0 <= theta & theta <= angles(2)+theta0)|...
         (angles(1)-2*pi+theta0 <= theta & theta <= angles(2)-2*pi+theta0)|...
         (angles(1)+2*pi+theta0 <= theta & theta <= angles(2)+2*pi+theta0));
    % Keep the valid world/pixel coordinates
    ROI{i} = p(ind,:);
    ROIpxl{i} = [X(ind) Y(ind)];
end

end