function [beta,mu] = est_gumbel(data)
%EST_GUMBEL Parameter estimates for Gumbel data.
%
% CALL: [beta,mu] = wgumbfit(data) 

%make sure it is a vector
data=data(:); 

start=6^(1/2)/pi*std(data); % Moment estimate of scale parameter a
beta = fzero(@(x) wgumbafit(x,data),start,optimset);
mu = -beta*log(mean(exp(-data/beta)));

function l=wgumbafit(a,data)
l=a-mean(data)+mean(data.*exp(-data/a))/mean(exp(-data/a));


