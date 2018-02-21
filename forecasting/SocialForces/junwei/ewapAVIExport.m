%% Script to generate an AVI file of energy visualization

if ~exist('D','var'), load ewap_dataset; [D Others] = seq2ewap(seq); end
addpath('SFfunctions');
% Set up avi file
aviobj = avifile('output.avi','fps',2.5,'videoname','ETH sequence');

% Set up parameters
datasets = fieldnames(seq);
dataset_id = 2;
% params = [0.130081 2.087902 2.327072 2.0732 1.461249 0.7304224];
params = [ 0.010883 2.646596 0.069719 2.496652 2.531520 0.353709 ];
% params = [ 0.185446 0.013518 2.821385 0.904783 1.441768 0.309408 ];
% params = [ 0.088017 0.301746 3.096017 2.173205 1.309891 0.349910 ];
% params = [ 0.025710 1.562468 2.109802 1.286950 2.285530 0.280542 ];
% params = [ 0.066759 1.572163 0.971813 1.395137 1.661488 0.392941 ];
Time = unique(D(D(:,1)==dataset_id,2));
for t = 1:length(Time)
    % Get image and data to draw
    img = read(seq.(datasets{dataset_id}).video,Time(t));
    ind = find(D(:,1)==dataset_id & D(:,2)==Time(t));
    % Draw
    [img,Phat] = ewapVisualize(img,seq.(datasets{dataset_id}).H,...
        D(ind,:),Others(ind),params);
    % Plot
    imshow(img,'Border','tight');
    hold on;
    Phat = [Phat ones(size(Phat,1),1)] / seq.(datasets{dataset_id}).H';
    plot(Phat(:,2)./Phat(:,3),Phat(:,1)./Phat(:,3),'r*');
    Plin = [D(ind,4:5)+0.4*D(ind,6:7) ones(size(Phat,1),1)] / seq.(datasets{dataset_id}).H';
    plot(Plin(:,2)./Plin(:,3),Plin(:,1)./Plin(:,3),'g+');
    Ptruth = [D(ind,8) D(ind,9) ones(size(Phat,1),1)] / seq.(datasets{dataset_id}).H';
    plot(Ptruth(:,2)./Ptruth(:,3),Ptruth(:,1)./Ptruth(:,3),'wo');
    hold off;
    legend({'LTA','LIN','Truth'});
    
    % Write
    aviobj = addframe(aviobj,gca);
end
close;
aviobj = close(aviobj);