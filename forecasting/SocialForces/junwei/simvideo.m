function []=simvideo(map)

k = map;                                    % which map to plot
cd /scr/alexr/SocialForces/

aviobj = VideoWriter(['simtrajEWAP' num2str(k) '.avi']);
open(aviobj);

file =['/scr/alexr/simtrajRdSF/simtrajRdSF' num2str(k) '.csv'];
f = importdata(file);
scale = 10;
for t=unique(f(:,3))'
    
    Vid = f(f(:,3)==t,:);
    
    plot(Vid(:,1)./scale,Vid(:,2)./scale,'bo');        % current agents' position
    
    %         hold on
    %         plot(Vid2(:,1)./scale,Vid2(:,2)./scale,'ro');
    %         hold on
    %         plot(Vid3(:,1)./scale,Vid3(:,2)./scale,'go');
    %

    hold off
    axis([-5 45 -5 45]);
    frame = getframe;
    writeVideo(aviobj,frame);
    
end

close(aviobj)

end