function [ Dtrain, Dtest ] = datasplit( D )
%DATASPLIT splits dataset to training/test subset by median of frames

% Split dataset to training/test subset
for did = 1:length(D)
    Dtrain(did).label = D(did).label;
    Dtest(did).label = D(did).label;
    Dtrain(did).video = D(did).video;
    Dtest(did).video = D(did).video;
    Dtrain(did).H = D(did).H;
    Dtest(did).H = D(did).H;
    Dtrain(did).obstacles = D(did).obstacles;
    Dtest(did).obstacles = D(did).obstacles;
    
    ind = D(did).observations(:,1)<median(D(did).observations(:,1));
    persons = unique(D(did).observations(ind,2));
    pind1 = arrayfun(@(k)any(D(did).persons(k,1)==persons),...
                    1:size(D(did).persons,1));
    Dtrain(did).persons = D(did).persons(pind1,:);        
    persons = unique(D(did).observations(~ind,2));
    pind2 = arrayfun(@(k)any(D(did).persons(k,1)==persons),...
                    1:size(D(did).persons,1));
    Dtest(did).persons = D(did).persons(pind2,:);
    
    Dtrain(did).observations = D(did).observations(ind,:);
    Dtest(did).observations = D(did).observations(~ind,:);
    fprintf('Dataset %s\n',D(did).label);
    fprintf('  Training / Test: %d + %d = %d observations\n',...
        nnz(ind),nnz(~ind),length(ind));
    fprintf('  Training / Test: %d / %d persons\n',nnz(pind1),nnz(pind2));
    
    Dtrain(did).destinations = D(did).destinations;
    Dtest(did).destinations = D(did).destinations;
end

end

