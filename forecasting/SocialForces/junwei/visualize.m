function visualize(dir,p)

ped = importpedstart(dir);

if nargin<2
    p = 1;
end
mx = min(ped(:,1))-10;
Mx = max(ped(:,1));
my = min(ped(:,2))-10;
My = max(ped(:,2));

for k=ped(:,3)'
    
    
    f = ped(ped(:,3)==k,1:2);
    plot(f(:,1),f(:,2),'o');
    %axis([mx Mx my My]);
    axis([0 400 0 400]);
    if p==1
        pause
    end
    pause(0.0005)
    hold off
end
