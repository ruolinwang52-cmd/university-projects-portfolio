%% Section 1.1 Load VOLVOA (Track A) Data and Visualization
clear; clc; close all;
load('VOLV0A.mat');     
T = length(VOLVOA);
dailyClose = zeros(T,1);
dates = datetime([VOLVOA.date], 'ConvertFrom', 'yyyymmdd').';   
for k = 1:T
    dailyClose(k,1) = VOLVOA(k).close; % Extract daily closing prices and dates
end

% Plot daily closing prices
figure;
plot(dates, dailyClose, 'b', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Daily Closing Price (VOLVO)');
title('VOLVO Daily Closing Prices (Track A)');
grid on;
%% Section 1.2 Compute and visualize daily log-returns
% Compute log returns from daily closing prices
r = diff(log(dailyClose));      % r_t = log(P_t) - log(P_{t-1})
rDates = dates(2:end);          
figure;
plot(rDates, r, 'k', 'LineWidth', 1); 
xlabel('Date');         %
ylabel('Log Return');
title('VOLVO Daily Log-Returns (Track A)');
grid on;
%% Section 1.3 EGARCH(1,1) with Gaussian innovations
% par_G = [mu, omega, alpha, beta, cLev]
mu0    = mean(r);  
omega0 = 0.0001;     
alpha0 = 0.1;       
beta0  = 0.8;       
cLev0  = 0.0001;       
theta0_G = [mu0; omega0; alpha0; beta0; cLev0];
[thetaG_hat, lnLG, VCG] = MLmax(@lnl_egarch_gauss, theta0_G, r);
seG = sqrt(diag(VCG));
CI_G_low  = thetaG_hat - 1.96*seG;
CI_G_high = thetaG_hat + 1.96*seG;
EGARCH_Gauss_Table = table(thetaG_hat, seG, CI_G_low, CI_G_high, ...
    'VariableNames', {'Estimate','SE','CI_low','CI_high'},  'RowNames', {'mu','omega','alpha','beta','cLev'});

disp('Gaussian EGARCH(1,1): estimates, SE, and 95% CI');
disp(EGARCH_Gauss_Table);
fprintf('Log-likelihood (EGARCH Gaussian) lnL = %.4f\n', lnLG);
%% Section 1.4 EGARCH(1,1) with heavy-tailed innovations (lnl_spec)
% par_S = [mu, omega, alpha, beta, cLev, cTail, fTail]
cTail0 = 0.2;   % initial tail parameter (0 = Gaussian)
fTail0 = 2;     % initial shape/smooth parameter
theta0_S = [mu0; omega0; alpha0; beta0; cLev0; cTail0; fTail0];
[thetaS_hat, lnLS, VCS] = MLmax(@lnl_egarch_spec, theta0_S, r);
seS = sqrt(diag(VCS));
CI_S_low  = thetaS_hat - 1.96*seS;
CI_S_high = thetaS_hat + 1.96*seS;
EGARCH_Spec_Table = table(thetaS_hat, seS, CI_S_low, CI_S_high, ...
    'VariableNames', {'Estimate','SE','CI_low','CI_high'},  'RowNames', {'mu','omega','alpha','beta','cLev','cTail','fTail'});

disp('Heavy-tailed EGARCH(1,1): estimates, SE, and 95% CI');
disp(EGARCH_Spec_Table);
fprintf('Log-likelihood (EGARCH heavy-tailed) lnL = %.4f\n', lnLS);
% We skip the EGARCH since the results from estimated parameter is not
% economicly meaningful (barely any leverage effect/ volatility clusting)
%% Section 1.3 GARCH(1,1) with Gaussian innovations
% par_G = [mu, omega, alpha, beta]
mu0    = mean(r);          
omega0 = 0.01*var(r);      
alpha0 = 0.05;             
beta0  = 0.9;             
theta0_G = [mu0; omega0; alpha0; beta0];
[thetaG_hat, lnLG, VCG] = MLmax(@lnl_garch_gauss, theta0_G, r);
seG = sqrt(diag(VCG));
CI_G_low  = thetaG_hat - 1.96*seG;
CI_G_high = thetaG_hat + 1.96*seG;
GARCH_Gauss_Table = table(thetaG_hat, seG, CI_G_low, CI_G_high, ...
    'VariableNames', {'Estimate','SE','CI_low','CI_high'}, ...
    'RowNames', {'mu','omega','alpha','beta'});

disp('Gaussian GARCH(1,1): estimates, SE, and 95% CI');
disp(GARCH_Gauss_Table);
fprintf('Log-likelihood (GARCH Gaussian) lnL = %.4f\n', lnLG);
%% Section 1.4 GARCH(1,1) with heavy-tailed innovations (lnl_spec)
% par_S = [mu, omega, alpha, beta, cTail, fTail]
cTail0 = 0.2;   % tail parameter (0 = Gaussian)
fTail0 = 2;     % shape/smooth parameter
theta0_S = [mu0; omega0; alpha0; beta0; cTail0; fTail0];
[thetaS_hat, lnLS, VCS] = MLmax(@lnl_garch_spec, theta0_S, r);
seS = sqrt(diag(VCS));
CI_S_low  = thetaS_hat - 1.96*seS;
CI_S_high = thetaS_hat + 1.96*seS;
GARCH_Spec_Table = table(thetaS_hat, seS, CI_S_low, CI_S_high, ...
    'VariableNames', {'Estimate','SE','CI_low','CI_high'},  'RowNames', {'mu','omega','alpha','beta','cTail','fTail'});

disp('Heavy-tailed GARCH(1,1): estimates, SE, and 95% CI');
disp(GARCH_Spec_Table);
fprintf('Log-likelihood (GARCH heavy-tailed) lnL = %.4f\n', lnLS);;
%% Section 1.5 Likelihood–ratio test: Gaussian vs Heavy–Tailed GARCH
kG = numel(thetaG_hat);  
kS = numel(thetaS_hat);   
LR  = 2*(lnLS - lnLG);    % LR statistic
df  = kS - kG;           % 2 more param
pLR = 1 - chi2cdf(LR, df);
fprintf('\nLikelihood–ratio test: Gaussian vs Heavy–Tailed GARCH(1,1)\n');
fprintf('  lnL (Gaussian)     = %8.4f\n', lnLG);
fprintf('  lnL (Heavy–tailed) = %8.4f\n', lnLS);
fprintf('  LR statistic       = %8.4f  (df = %d)\n', LR, df);
fprintf('  p–value            = %8.4g\n\n', pLR);


%Reject Gaussian one, the GARCH 1-1 heavy tailed one works a lot better 





%% Section 1.6 95% CI for in-sample and one future beyond sample
muS    = thetaS_hat(1);
omegaS = thetaS_hat(2);
alphaS = thetaS_hat(3);
betaS  = thetaS_hat(4);
cTailS = thetaS_hat(5);
fTailS = thetaS_hat(6);
T = length(r);
epsS = r - muS;
hS = zeros(T,1);
hS(1) = var(r);      
for t = 2:T
    hS(t) = omegaS + alphaS*epsS(t-1)^2 + betaS*hS(t-1);
end
alphaLevel = 0.05;
pUpper = 1 - alphaLevel/2;    % 0.975
qHT = lnl_spec_quantile(pUpper, cTailS, fTailS);
fprintf('Heavy–tailed 97.5%% quantile q_0.975 = %.4f\n', qHT);
% 95% in-sample next-ahead prediction
T = length(r);
condMean = muS * ones(T,1);
condStd  = sqrt(hS);
lowerHT = condMean - qHT * condStd;
upperHT = condMean + qHT * condStd;
% Drop t=1 
rDatesEff   = rDates;
rEff        = r;
lowerHT_eff = lowerHT;
upperHT_eff = upperHT;
figure;
plot(rDatesEff, rEff, 'k-', 'LineWidth', 1); hold on;
plot(rDatesEff, lowerHT_eff, 'r--', 'LineWidth', 1);
plot(rDatesEff, upperHT_eff, 'r--', 'LineWidth', 1);
xlabel('Date');
ylabel('Log return');
title('95% One-step-ahead Prediction Intervals (Heavy-tailed GARCH(1,1))');
legend('Realized return','Lower 95%','Upper 95%','Location','best');
grid on;
eps_last = epsS(end);
h_last   = hS(end);%Forecast 1-step ahead (T+1)
h_Tplus1 = omegaS + alphaS*eps_last^2 + betaS*h_last;
mean_Tplus1 = muS;
std_Tplus1  = sqrt(h_Tplus1);
PI_Tplus1 = [ mean_Tplus1 - qHT*std_Tplus1,mean_Tplus1 + qHT*std_Tplus1 ];
fprintf('95%% PI for r_{T+1}: [%.4f, %.4f]\n', PI_Tplus1(1), PI_Tplus1(2));



% We see the bands widen during volatile periods with presistance. 





%% Section 1.7 Christoffersen Conditional Coverage Test (LR_CC)

hits = (r < lowerHT_eff) | (r > upperHT_eff);   %   1 = violation
I = hits(:);   
T = length(I);
% Count transitions
n00 = sum(I(1:end-1)==0 & I(2:end)==0);
n01 = sum(I(1:end-1)==0 & I(2:end)==1);
n10 = sum(I(1:end-1)==1 & I(2:end)==0);
n11 = sum(I(1:end-1)==1 & I(2:end)==1);
pi_hat = mean(I);% Unconditional coverage violation frequency
% Markov transition probabilities
pi01 = n01 / (n00 + n01);  % P(I_t =1 | I_{t-1}=0)
pi11 = n11 / (n10 + n11);  % P(I_t =1 | I_{t-1}=1)
% Likelihood under independence (correct coverage)
L1 = (1 - pi_hat)^(n00 + n10) * (pi_hat)^(n01 + n11);
% Likelihood under 1st-order Markov chain
L2 = (1 - pi01)^n00 * (pi01)^n01 * (1 - pi11)^n10 * (pi11)^n11;
% LR_CC statistic
LR_CC = -2*(log(L1) - log(L2));
p_CC = 1 - chi2cdf(LR_CC, 2);   % df = 2
fprintf('\nChristoffersen Conditional Coverage Test (LR_CC)\n');
fprintf('  n00 = %d, n01 = %d, n10 = %d, n11 = %d\n', n00, n01, n10, n11);
fprintf('  LR_CC = %.4f  (df = 2)\n', LR_CC);
fprintf('  p-value = %.4f\n\n', p_CC);
if p_CC < 0.05
    disp('Reject: Interval forecasts FAIL the conditional coverage test.');
else
    disp('Do NOT reject: Interval forecasts PASS the conditional coverage test.');
end

%% Section 1.8  Visualize all intraday (10-minute) prices

Tdays  = numel(VOLVOA);             
nIntra = numel(VOLVOA(1).price);  
allTimes  = NaT(Tdays*nIntra,1);    
allPrices = zeros(Tdays*nIntra,1); 
idx = 0;
for k = 1:Tdays
    baseDate = datetime(VOLVOA(k).date, 'ConvertFrom','yyyymmdd');
    timeNum = VOLVOA(k).time;    
    hh = floor(timeNum/100);
    mm = timeNum - 100*hh;
    thisTimes  = baseDate + hours(hh) + minutes(mm);
    thisPrices = VOLVOA(k).price;
    allTimes(idx+1:idx+nIntra)  = thisTimes;
    allPrices(idx+1:idx+nIntra) = thisPrices;
    idx = idx + nIntra;
end
figure;
plot(allTimes, allPrices, 'k-', 'LineWidth', 0.8);
xlabel('Date & Time');
ylabel('Intraday Price (10-minute)');
title('VOLVO Intraday 10-minute Prices (Track A)');
grid on;

%% Section 1.9 Compute log returns 
intraReturns = [];    %  collect all intraday returns (excluding first of each day)
intraTimes    = [];   
idx = 0;
for k = 1:Tdays

    P = VOLVOA(k).price;   
    tnum = VOLVOA(k).time; 
    baseDate = datetime(VOLVOA(k).date,'ConvertFrom','yyyymmdd');
    hh = floor(tnum/100);
    mm = tnum - 100*hh;
    dayTimes = baseDate + hours(hh) + minutes(mm);
    dayRet  = diff(log(P));
    dayTime = dayTimes(2:end);
    intraReturns = [intraReturns; dayRet];
    intraTimes   = [intraTimes;   dayTime];

end
figure;
plot(intraTimes, intraReturns, 'b-', 'LineWidth', 0.7);
xlabel('Datetime');
ylabel('Intraday Log Return (10-min)');
title('Intraday 10-minute Log Returns');
grid on;
%% Section 1.10 Compute daily BPV and estimate GARCH-X(1,1) with heavy-tailed noise

% Compute BVP
Tdays = length(VOLVOA);
BPV = zeros(Tdays,1);
for k = 1:Tdays
    P = VOLVOA(k).price;             
    rIntra = diff(log(P));            
% Bipower variation for day k
    BPV(k) = (pi/2) * sum( abs(rIntra(2:end)) .* abs(rIntra(1:end-1)) );
end

X = BPV(1:end-1);         
dataGX = [r, X];         
mu0     = mean(r);
omegaFixed = 0.0001;
alpha0  = 0.05;        
beta0   = 0.9;
gamma0  = 0.0001;               % Tryed bounch of initial guesses, especially gamma, only this did not stuck outside constrains
cTail0  = 0.2;            
fTail0  = 2;              
theta0_GX = [mu0; alpha0; beta0; gamma0; cTail0; fTail0];
[thetaGX_hat, lnLGX, VCGX] = MLmax(@lnl_garchx_spec, theta0_GX, dataGX);

disp('Heavy-tailed GARCH-X(1,1) parameters [mu, omega(fixed), alpha, beta, gamma, cTail, fTail]:');
disp([thetaGX_hat(1); omegaFixed; thetaGX_hat(2:end)]);
fprintf('Log-likelihood (GARCH-X heavy-tailed) lnL = %.4f\n', lnLGX);

% 95%  CIs for estimated parameters 
seGX = sqrt(diag(VCGX));                 % SEs 
CI_low_GX  = thetaGX_hat - 1.96*seGX;
CI_high_GX = thetaGX_hat + 1.96*seGX;

paramNamesGX = {'mu','alpha','beta','gamma','cTail','fTail'}';
T_GX = table(thetaGX_hat, seGX, CI_low_GX, CI_high_GX, 'VariableNames', {'Estimate','SE','CI_low','CI_high'},'RowNames', paramNamesGX);

disp('95% Wald CIs for estimated GARCH-X parameters (omega is fixed, no CI):');
disp(T_GX);

fprintf('omega (fixed) = %.6g (no SE/CI because it is fixed)\n', omegaFixed);

%% Section 1.11 95% one-step-ahead prediction intervals for daily log-returns (GARCH-X)


muGX     = thetaGX_hat(1);
alphaGX  = thetaGX_hat(2);
betaGX   = thetaGX_hat(3);
gammaGX  = thetaGX_hat(4);
cTailGX  = thetaGX_hat(5);
fTailGX  = thetaGX_hat(6);
omegaGX  = omegaFixed;           % fixed small intercept
T = length(r);                   
X = BPV(1:end-1);                
epsGX = r - muGX;              
hGX   = zeros(T,1);
hGX(1) = var(r);               
for t = 2:T
    hGX(t) = omegaGX + alphaGX*epsGX(t-1)^2 + betaGX*hGX(t-1) + gammaGX*X(t-1);
    if hGX(t) <= 0
        error('Negative conditional variance encountered at t=%d', t);
    end
end
% 97.5% quantile of the heavy-tailed innovation distribution, same as in Garch(1,1)
alphaLevel = 0.05;
pUpper = 1 - alphaLevel/2;   % 0.975
qHT_GX = lnl_spec_quantile(pUpper, cTailGX, fTailGX);
fprintf('GARCH-X heavy-tailed q_0.975 = %.4f\n', qHT_GX);
% In-sample 95% one-step-ahead prediction intervals for daily returns
condMeanGX = muGX * ones(T,1);
condStdGX  = sqrt(hGX);
lowerGX = condMeanGX - qHT_GX * condStdGX;
upperGX = condMeanGX + qHT_GX * condStdGX;
figure;
plot(rDates, r, 'k-', 'LineWidth', 1); hold on;
plot(rDates, lowerGX, 'b--', 'LineWidth', 1);
plot(rDates, upperGX, 'b--', 'LineWidth', 1);
xlabel('Date');
ylabel('Daily log-return');
title('95% one-step-ahead PIs – Heavy-tailed GARCH-X(1,1) with BPV');
legend('Realized return','Lower 95%','Upper 95%','Location','best');
grid on;
% One-step-ahead PI for r_{T+1} (beyond the sample)
eps_lastGX = epsGX(end);
h_lastGX   = hGX(end);
X_last     = X(end);
h_Tplus1_GX   = omegaGX + alphaGX*eps_lastGX^2 + betaGX*h_lastGX + gammaGX*X_last;
mean_Tplus1GX = muGX;
std_Tplus1GX  = sqrt(h_Tplus1_GX);
PI_Tplus1_GX = [ mean_Tplus1GX - qHT_GX*std_Tplus1GX, mean_Tplus1GX + qHT_GX*std_Tplus1GX ];
fprintf('GARCH-X: 95%% PI for r_{T+1}: [%.4f, %.4f]\n', PI_Tplus1_GX(1), PI_Tplus1_GX(2));

%% Section 1.12 Christoffersen conditional coverage test for GARCH-X intervals
alphaLevel = 0.05;          % nominal 95% PI -> 5% exceedance probability
p0 = alphaLevel;            
% Indicator of violations: 1 if r_t outside [lowerGX, upperGX], 0 otherwise
violGX = (r < lowerGX) | (r > upperGX);   % T x 1 logical
I = violGX(:);                          
T_I = length(I);
N1 = sum(I);             
N0 = T_I - N1;           
pi_hat = N1 / T_I;      
logL_uc1 = N1*log(pi_hat) + N0*log(1 - pi_hat);     
logL_uc0 = N1*log(p0)      + N0*log(1 - p0);       
LR_UC    = -2*(logL_uc0 - logL_uc1);
p_UC     = 1 - chi2cdf(LR_UC, 1);
n00 = sum( I(1:end-1)==0 & I(2:end)==0 );
n01 = sum( I(1:end-1)==0 & I(2:end)==1 );
n10 = sum( I(1:end-1)==1 & I(2:end)==0 );
n11 = sum( I(1:end-1)==1 & I(2:end)==1 );
pi01_hat = n01 / (n00 + n01);    % P(I_t=1 | I_{t-1}=0)
pi11_hat = n11 / (n10 + n11);    % P(I_t=1 | I_{t-1}=1)
% Likelihood under 2-state Markov alternative
logL_ind1 = n00*log(1 - pi01_hat) + n01*log(pi01_hat) + n10*log(1 - pi11_hat); %removed n11 * log(pi11_hat) becomes 0 * log(0) = NaN
logL_ind0 = (n00 + n10)*log(1 - pi_hat) + (n01 + n11)*log(pi_hat);
LR_IND = -2*(logL_ind0 - logL_ind1);
p_IND  = 1 - chi2cdf(LR_IND, 1);
LR_CC = LR_UC + LR_IND;
p_CC  = 1 - chi2cdf(LR_CC, 2);   % df = 2
fprintf('\nChristoffersen tests for GARCH-X heavy-tailed intervals (alpha = %.2f)\n', alphaLevel);
fprintf('  Unconditional coverage: LR_UC  = %6.4f,  p = %6.4f\n', LR_UC,  p_UC);
fprintf('  Independence:           LR_IND = %6.4f,  p = %6.4f\n', LR_IND, p_IND);
fprintf('  Conditional coverage:   LR_CC  = %6.4f,  p = %6.4f\n\n', LR_CC, p_CC);
if p_CC < 0.05
    fprintf('=> Reject conditional coverage at 5%%: GARCH-X intervals FAIL the CC test.\n');
else
    fprintf('=> Do NOT reject conditional coverage at 5%%: GARCH-X intervals PASS the CC test.\n');
end

%% Likelihood–ratio test: GARCH vs GARCH-X (BPV)

lnL_GARCH  = lnLS;     
lnL_GARCHX = lnLGX;    
LR = 2 * (lnL_GARCHX - lnL_GARCH);
% Degrees of freedom 
df = 1;
% p-value
p_LR = 1 - chi2cdf(LR, df);
fprintf('\nLikelihood–Ratio test: GARCH vs GARCH-X\n');
fprintf('  H0: gamma = 0 (no BPV effect)\n');
fprintf('  LR statistic = %.4f\n', LR);
fprintf('  df = %d\n', df);
fprintf('  p-value = %.4f\n', p_LR);

if p_LR < 0.05
    fprintf('=> Reject H0 at 5%%: BPV significantly improves the model.\n');
else
    fprintf('=> Do NOT reject H0 at 5%%: No significant improvement from BPV.\n');
end



% Helper functions 
function l = lnl_egarch_gauss(par, r)

%   Model:
%       r_t = mu + eps_t
%       eps_t = sqrt(h_t) * z_t,   z_t ~ N(0,1)
%       log h_t = omega + alpha * ( |z_{t-1}| - cLev*z_{t-1} ) + beta  * log h_{t-1}
mu    = par(1);
omega = par(2);
alpha = par(3);
beta  = par(4);
cLev  = par(5);
T = length(r);
  
l = NaN(T,1);   
%Parameter constraints
if (beta <= 0) || (beta >= 1) || (alpha <= 0) || (abs(cLev) >= 1)
    return; 
end
eps = r - mu;
lnh = zeros(T,1);
h   = zeros(T,1);
z   = zeros(T,1);
% Initial values
lnh(1) = log(var(r));   
h(1)   = exp(lnh(1));
z(1)   = 0;             

for t = 2:T
    f_prev = abs(z(t-1)) - cLev*z(t-1);  % leverage function f(z)
    lnh(t) = omega + alpha*f_prev + beta*lnh(t-1);
    h(t)   = exp(lnh(t));
    z(t)   = eps(t) / sqrt(h(t));        % standardized residual
end

% Gaussian log-likelihood contributions
l = -0.5*( log(2*pi) + log(h) + (eps.^2)./h );
end
function l = lnl_egarch_spec(par, r)
% lnl_egarch_spec  Log-likelihood for EGARCH(1,1) with heavy-tailed noise using Magnus Wiktorsson's lnl_spec.m
mu    = par(1);
omega = par(2);
alpha = par(3);
beta  = par(4);
cLev  = par(5);   % leverage parameter in f(z)
cTail = par(6);   % tail parameter for lnl_spec
fTail = par(7);   % shape parameter for lnl_spec
T = length(r);
     

l = NaN(T,1);     

% Parameter constraints 
if (beta <= 0) || (beta >= 1) || (alpha <= 0) || (abs(cLev) >= 1) || (cTail < 0) || (cTail >= 1) || (fTail <= 0)
    return; 
end
eps = r - mu;
lnh = zeros(T,1);
h   = zeros(T,1);
z   = zeros(T,1);
% Initial values
lnh(1) = log(var(r));
h(1)   = exp(lnh(1));
z(1)   = 0;
for t = 2:T
    f_prev = abs(z(t-1)) - cLev*z(t-1);
    lnh(t) = omega + alpha*f_prev + beta*lnh(t-1);
    h(t)   = exp(lnh(t));
    z(t)   = eps(t) / sqrt(h(t));
end
% Now use lnl_spec for the innovation density
s_t   = sqrt(h);                 
parHF = [cTail; fTail];          
l = lnl_spec(parHF, s_t, eps);   
end
function l = lnl_garch_gauss(par, r)
% lnl_garch_gauss  Log-likelihood contributions for Gaussian GARCH(1,1)
%   Model:
%       r_t = mu + eps_t
%       eps_t = sqrt(h_t) * z_t,   z_t ~ N(0,1)
%       h_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}

mu    = par(1);
omega = par(2);
alpha = par(3);
beta  = par(4);

T = length(r);
l = NaN(T,1);   
% Parameter constraints 

if (omega <= 0) || (alpha < 0) || (beta < 0) || (alpha + beta >= 1)
    return;     
end
eps = r - mu;
h = zeros(T,1);
h(1) = var(r);  
for t = 2:T
    h(t) = omega + alpha*eps(t-1)^2 + beta*h(t-1);
end
l = -0.5*( log(2*pi) + log(h) + (eps.^2)./h );
end
function l = lnl_garch_spec(par, r)

% lnl_garch_spec  Log-likelihood for GARCH(1,1) with heavy-tailed noise using Magnus Wiktorsson's lnl_spec.m
%   Volatility recursion:
%       h_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}
%   Innovations density:
%       eps_t | F_{t-1} ~ lnl_spec( cTail, fTail, scale = sqrt(h_t) )

mu    = par(1);
omega = par(2);
alpha = par(3);
beta  = par(4);
cTail = par(5);   % tail parameter (0<=c<1)
fTail = par(6);   % shape parameter (f>0)
T = length(r);
l = NaN(T,1);
% Parameter constraints 
if (omega <= 0) || (alpha < 0) || (beta < 0) || (alpha + beta >= 1)
    return;
end
% Parameter constraints for lnl_spec 
if (cTail < 0) || (cTail >= 1) || (fTail <= 0)
    return;
end
eps = r - mu;
h = zeros(T,1);
h(1) = var(r);
for t = 2:T
    h(t) = omega + alpha*eps(t-1)^2 + beta*h(t-1);
end

% Use lnl_spec for the innovation density:
s_t   = sqrt(h);          % conditional std dev
parHF = [cTail; fTail];   % tail parameters
l = lnl_spec(parHF, s_t, eps);    
end
function q = lnl_spec_quantile(p, cTail, fTail)

% Returns the p-quantile of the standardized lnl_spec distribution using numerical integration + root finding.

    parHF = [cTail; fTail];
% CDF of lnl_spec with scale s = 1
    F = @(x) integral(@(u) exp(lnl_spec(parHF,1,u)), -10, x,'ArrayValued', true);
% Solve F(q) = p
    obj = @(x) F(x) - p;
    q = fzero(obj, 2);   % initial guess 
end
function l = lnl_garchx_spec(par, data)
% lnl_garchx_spec  Log-likelihood for GARCH-X(1,1) with heavy-tailed noise using Magnus Wiktorsson's lnl_spec.m
% par  = [mu, alpha, beta, gamma, cTail, fTail]
%   data = [r, X],  where
%   r = daily log-returns   (T x 1)
%    X = BPV regressor       (T x 1)
%   omegafixed=0.0001
%   h_t = omega(we set as a constant,otherwise estimation is not within constrain, since omega is very small anyways in finance data of prices) + alpha*eps_{t-1}^2 + beta*h_{t-1} + gamma*X_{t-1}
%   eps_t | F_{t-1} ~ lnl_spec(cTail,fTail, scale = sqrt(h_t))

    r = data(:,1);
    X = data(:,2);
    mu    = par(1);
    
    alpha = par(2);
    beta  = par(3);
    gamma = par(4);
    cTail = par(5);
    fTail = par(6);
    T = length(r);
    l = NaN(T,1);            
    
    eps = r - mu;
    h = zeros(T,1);
    h(1) = var(r);     % initial variance,% We drop the parameter constrains here since they are very restricive and also becuase our parameters estimated are within the constrain. 
    for t = 2:T
        h(t) = 0.0001 + alpha*eps(t-1)^2 + beta*h(t-1) + gamma*X(t-1);
        if h(t) <= 0
            l = NaN(T,1);
            return;
        end
    end
    s_t   = sqrt(h);          
    parHF = [cTail; fTail];   
    l = lnl_spec(parHF, s_t, eps);  
end