function num_frame_max = readTraj(trackingCsvFile,parameters)
% Calculate istantaneus Heatmap (LoS)
% Usage:
% heatMap=displayHeatmap(trackingCsvFile)
%
% Visiosafe 2013 ?
% Author: Luigi Bagnato, luigi.bagnato@visiosafe.com

if ~exist('trackingCsvFile','var')
    %trackingCsvFile='..//toy//0//dynamic//posN.csv';
    trackingCsvFile='posN.csv';
end

if ~exist('parameters','var')
    parameters=struct;
end



%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LosNormFac=1.3/(.055^2); %This is a parameter TO SET
if ~isfield(parameters,'pixel_size')
    pix2mmFac=1/55;
else
    pix2mmFac=1/(parameters.pixel_size*1000);
end

if ~isfield(parameters,'time_step')
    time_step=100; %in millisec
else
    time_step=parameters.time_step;
end

if ~isfield(parameters,'duration')
    parameters.duration=1; %in minutes
end

if ~isfield(parameters,'pixel_size')
    image_name='plan_piwest_h.png';
    base_image=imread(image_name);
else
    base_image=parameters.base_image;
end


isVerbose=true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load q2 image

[nrows,ncols,nchannels]=size(base_image);

% Load data
f_id = fopen(trackingCsvFile);

M = textscan(f_id,'%d %d %d %d %d','delimiter',',');

x_pos = M{1};
y_pos = M{2};
date_time = M{3};
tracklet_id = M{4}; % not used


num_frame_max = 600;%max(date_time)

occurrency_vec=zeros(nrows*ncols,1);

% we iterate over time
frame_counter=1;
while frame_counter<num_frame_max
    
    index_time = (date_time == frame_counter);
    
    tmp_pos_x = x_pos(index_time);
    tmp_pos_y = y_pos(index_time);
    
    index_x = min(max(ceil(tmp_pos_x),1),ncols);
    index_y = min(max(ceil(tmp_pos_y),1),nrows);
    
    
    lenV = nnz(index_time);
    
    lin_index=(index_x-1)*nrows + index_y;
    
    %occurrency_vec=zeros(nrows*ncols,1);
    for i_o=1:numel(lin_index)
        %        occurrency_vec(lin_index(i_o)) = occurrency_vec(lin_index(i_o)) + 1;
        occurrency_vec(lin_index(i_o)) =  1;
    end
    %
    
    %%%%%%
    
    
       if isVerbose
           heatMap_currentframe = zeros(nrows,ncols);
           heatMap_currentframe(:) = occurrency_vec;

          %imagesc(base_image + heatMap_mat(:,:,frame_counter));axis image ;
           imshow( heatMap_currentframe, [0 1]);axis image ;
           %HeatMap( heatMap_currentframe);
           drawnow;
           %pause;
       end
    %%%%%%%%%
    
    
    
    frame_counter = frame_counter + 1;
end

heatMap_currentframe = zeros(nrows,ncols);
heatMap_currentframe(:) = occurrency_vec;


if isVerbose
    figure(1);
    %imagesc(base_image + heatMap_mat(:,:,frame_counter));axis image ;
    imshow( heatMap_currentframe, [0 1]);axis image ;
    %HeatMap( heatMap_currentframe);
    title(sprintf('Frame %i',frame_counter)); drawnow;
    
    %pause;
end
