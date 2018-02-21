function [C, order] = myconfusionmat(g, ghat)
%MYCONFUSIONMAT naive implementation to calculate confusion matrix
% This is to avoid version issue of statistics toolbox
% confusionmat() is only supported after ver 7
    order = unique([g;ghat]);
    C = zeros(length(order),length(order));
    for i = 1:length(g)
        C(order==g(i),order==ghat(i)) = C(order==g(i),order==ghat(i)) + 1;
    end
end