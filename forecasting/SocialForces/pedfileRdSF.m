clear all
close all


matfiles = dir(fullfile('/scr', 'alexr', 'simtrajRdSF', '*.csv'));
g = [];
for i=1:length(matfiles)
    g = [g str2double(matfiles(i).name(12:end-4))];
end


% matfiles = dir(fullfile('/scr', 'alexr', 'ped_fileRdSF', '*.csv'));
% 
% dd =[];
% for i=1:length(matfiles)
%     dd = [dd str2double(matfiles(i).name(9:end-4))]; %#ok<*AGROW>
% end
% 
% for k=1:length(dd)
%     a=find(g==dd(k));
%     g(a)=[];
% end



for k=sort(g)
    try %#ok<TRYNC>
        clear T P
        cd /scr/alexr/simtrajRdSF/
        fprintf([num2str(k) '\n']);
        T = importdata(['simtrajRdSF' num2str(k) '.csv']);
        P=[];
        for i=unique(T(:,4))'
            Ti = T(T(:,4)==i,:);
            if size(Ti,1)>=5
                P(i,1:2) = [1 1];      % fake origin
                P(i,3) = Ti(1,3);      % fake dest
                P(i,4) = norm((Ti(5,[1 2])-Ti(1,[1 2]))./5);     % first velocity
                P(i,5) = 1;    %class
                P(i,[6 7]) = Ti(1,[1 2]);    % first position
                P(i,8:9)=Ti(end,[1 2]);   %#ok<*SAGROW> % end position
                P(i,10) = Ti(end,3); % end frame
                P(i,11) = norm((Ti(end,[1 2])-Ti(1,[1 2]))./size(Ti,1)); % likely speed
            else
                T(T(:,4)==i,:)=[];
            end
            
        end
        cd /scr/alexr/ped_fileRdSF/
        csvwrite(['ped_file' num2str(k) '.csv'],P);
       
    end
    try %#ok<TRYNC>
        clear T P
        cd /scr/alexr/simtrajRdSF/
        fprintf([num2str(k) '\n']);
        T = importdata(['simtrajRdSF' num2str(k) '.csv']);
        P=[];i=0;
        len = 10;
        for j=unique(T(:,4))'
            Ti = T(T(:,4)==j,:);
            if size(Ti,1)>=len
                P(i+1:i+len,1:2) = repmat([1 1],len,1);      % fake origin + desst
                P(i+1:i+len,3) = Ti(1:len,3);      % first frames
                P(i+1:i+len,4) = repmat(norm((Ti(len,[1 2])-Ti(1,[1 2]))./len),len,1);     % first velocity
                P(i+1:i+len,5) = repmat(len,1);    %class
                P(i+1:i+len,[6 7]) = Ti(1:len,[1 2]);    % first position
                P(i+1:i+len,8:9)=repmat(Ti(end,[1 2]),len,1);   %#ok<*SAGROW> % end position
                P(i+1:i+len,10) = repmat(Ti(end,3),len,1); % end frame
                P(i+1:i+len,11) = repmat(norm((Ti(end,[1 2])-Ti(1,[1 2]))./size(Ti,1)),len,1); % likely speed
                P(i+1:i+len,12)=repmat(j,len,1); %id 
            else
                T(T(:,4)==j,:)=[];
            end
            i=i+5;
            
            
        end
        P(P(:,12)==0,:)=[];
        cd /scr/alexr/ped_fileSFtoIGP/
        csvwrite(['ped_file' num2str(k) '.csv'],P);
        
   end
    
end
% 
% 
% 
% for k=sort(g)
%    try %#ok<TRYNC>
%         clear T P
%         cd /scr/alexr/simtrajRdSF/
%         fprintf([num2str(k) '\n']);
%         T = importdata(['simtrajRdSF' num2str(k) '.csv']);
%         P=[];i=0;
%         for j=unique(T(:,4))'
%             Ti = T(T(:,4)==j,:);
%             if size(Ti,1)>=5
%                 P(i+1:i+5,1:2) = repmat([1 1],5,1);      % fake origin + desst
%                 P(i+1:i+5,3) = Ti(1:5,3);      % first frames
%                 P(i+1:i+5,4) = repmat(norm((Ti(5,[1 2])-Ti(1,[1 2]))./5),5,1);     % first velocity
%                 P(i+1:i+5,5) = repmat(5,1);    %class
%                 P(i+1:i+5,[6 7]) = Ti(1:5,[1 2]);    % first position
%                 P(i+1:i+5,8:9)=repmat(Ti(end,[1 2]),5,1);   %#ok<*SAGROW> % end position
%                 P(i+1:i+5,10) = repmat(Ti(end,3),5,1); % end frame
%                 P(i+1:i+5,11) = repmat(norm((Ti(end,[1 2])-Ti(1,[1 2]))./size(Ti,1)),5,1); % likely speed
%                 P(i+1:i+5,12)=repmat(j,5,1); %id 
%             else
%                 T(T(:,4)==j,:)=[];
%             end
%             i=i+5;
%             
%             
%         end
%         P(P(:,12)==0,:)=[];
%         cd /scr/alexr/ped_fileSFtoIGP/
%         csvwrite(['ped_file' num2str(k) '.csv'],P);
%         
%    end
% end
