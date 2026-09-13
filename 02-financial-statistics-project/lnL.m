function l=lnL(theta,X)
% Output must be a column vector

% Extract parameters
mu=theta(1);
sigma2=theta(2)^2;

l=-.5*log(2*pi*sigma2)-0.5*((X-mu).^2)/sigma2