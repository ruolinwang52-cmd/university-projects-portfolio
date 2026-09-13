%% Question 2  MC pricing of a European call 

clear; clc; close all;

%Parameters
S0    = 90;
K     = 80;
r     = 0.01;
sigma = 0.6;
T     = 1.0;

Ns = [1e4 1e5 1e6];                         
disc = exp(-r*T);

%BS model option value
function [d1,d2] = d1d2_bs(S0,K,r,sigma,T)
    vsqrtT = sigma*sqrt(T);
    d1 = (log(S0/K) + (r + 0.5*sigma^2)*T)/vsqrtT;
    d2 = d1 - vsqrtT;
end  
[d1,d2] = d1d2_bs(S0,K,r,sigma,T);
C_BS = S0*normcdf(d1) - K*disc*normcdf(d2);
fprintf('Black–Scholes price: %.6f\n\n', C_BS);

% Comparison storage
results = table('Size',[length(Ns) 10],  'VariableTypes', repmat("double",1,10), 'VariableNames', {'N','CrudePrice','CrudeError','CrudeSD', ...
                      'AntitheticPrice','AntitheticError','AntitheticSD', 'ControlPrice','ControlError','ControlSD'});

row = 1;

%MC Estimator 
for N = Ns

    % Crude MC
    Z  = randn(N,1);
    ST = S0 .* exp((r-0.5*sigma^2)*T + sigma*sqrt(T).*Z);
    Y  = disc*max(ST - K, 0);
    crudePrice   = mean(Y);
    crudeSD_pay  = std(Y,1);            % SD of payoffs
    crudeSD_est  = crudeSD_pay / sqrt(N);  % SD of estimated price
    crudeErr     = abs(crudePrice - C_BS);

    % Antithetic Variates
    Z   = randn(N,1);
    ST1 = S0 .* exp((r-0.5*sigma^2)*T + sigma*sqrt(T).*Z);
    ST2 = S0 .* exp((r-0.5*sigma^2)*T - sigma*sqrt(T).*Z);
    Yant = 0.5*(disc*max(ST1 - K, 0) + disc*max(ST2 - K, 0));
    antitheticPrice  = mean(Yant);
    antitheticSD_pay = std(Yant,1);
    antitheticSD_est = antitheticSD_pay / sqrt(N);   % SD of estimated price
    antitheticErr    = abs(antitheticPrice - C_BS);

    % Control Variate
    Z  = randn(N,1);
    ST = S0 .* exp((r-0.5*sigma^2)*T + sigma*sqrt(T).*Z);
    Y  = disc*max(ST - K, 0);
    X  = disc*ST;
    beta = sum((X - mean(X)).*(Y - mean(Y))) / sum((X - mean(X)).^2);
    Ycv   = Y - beta * (X - S0);
    controlPrice   = mean(Ycv);
    controlSD_pay  = std(Ycv,1);
    controlSD_est  = controlSD_pay / sqrt(N);        % SD of estimated price
    controlErr     = abs(controlPrice - C_BS);

    results(row,:) = {N, ...
        crudePrice,      crudeErr,      crudeSD_est, ...
        antitheticPrice, antitheticErr, antitheticSD_est, ...
        controlPrice,    controlErr,    controlSD_est};

    row = row + 1;
end

disp('Comparison of methods vs. Black–Scholes (SD columns = SD of estimated price):')
disp(results)

% Sometimes by luck, the crude estimate lands very close to the Black–Scholes price, 
% but this doesn’t mean it’s better.



%% OTM call pricing crude MC vs IS, we have that the probability that S_T > K is very small. Crude MC will waste most samples at 0 
clear; clc; close all;
S0    = 50;
K     = 150;
r     = 0.05;
sigma = 0.4;
T     = 1.0;
disc  = exp(-r*T);

%Sample sizes
Ns = [1e4 1e5 1e6];

%BS function and price 
function [C, d1, d2] = bs_call_price(S0,K,r,sigma,T)
    vsqrtT = sigma*sqrt(T);
    d1 = (log(S0/K) + (r + 0.5*sigma^2)*T) / vsqrtT;
    d2 = d1 - vsqrtT;
    C  = S0*normcdf(d1) - K*exp(-r*T)*normcdf(d2);
end
[bsPrice, d1, d2] = bs_call_price(S0,K,r,sigma,T);
fprintf('Black–Scholes price: %.8f\n', bsPrice);

% OTM threshold z* : payoff positive only if Z > zStar，if we dont use IS,
% then we waste a lot samples on the negative payoff. 
zStar = (log(K/S0) - (r - 0.5*sigma^2)*T)/(sigma*sqrt(T));
fprintf('OTM threshold z* = %.4f\n\n', zStar);


muShift = zStar;    % Shift of original distribution f, can tune around. 
results = table('Size',[length(Ns) 6], 'VariableTypes', repmat("double",1,6),'VariableNames', {'N','CrudePrice','CrudeSDest', ...
                                     'ISPrice','ISSDest','muShift'});

% MC estimator payoff
payoff = @(Z) max(S0*exp((r-0.5*sigma^2)*T + sigma*sqrt(T).*Z) - K, 0);

for i = 1:numel(Ns)
    N = Ns(i);

    %Crude MC
    Z  = randn(N,1);                  
    Yc = disc * payoff(Z);           
    crudePrice  = mean(Yc);
    crudeSDest  = std(Yc,1)/sqrt(N); 
    % IS
    mu = muShift;
    Zt = mu + randn(N,1);             % proposal q: N(mu,1)
    L  = exp(-mu*Zt + 0.5*mu^2);      % Correct ratio 
    Yi = disc * payoff(Zt) .* L;      
    isPrice   = mean(Yi);
    isSDest   = std(Yi,1)/sqrt(N);

    results{i, :} = [N, crudePrice, crudeSDest, isPrice, isSDest, mu];
end

disp('Crude MC vs Importance Sampling (SDest = SD of estimated price)')
disp(results) 
zGrid = linspace(zStar-3, zStar+6, 1200).';   
h  = payoff(zGrid);                          
phi    = (1/sqrt(2*pi))*exp(-0.5*zGrid.^2);
phiMu  = (1/sqrt(2*pi))*exp(-0.5*(zGrid-muShift).^2);
Lgrid  = phi ./ phiMu;                      
hL     = h .* Lgrid;

figure('Color','w'); 
subplot(1,2,1);
semilogy(zGrid, max(h,1e-300), 'b-', 'LineWidth',1.2); hold on;
semilogy(zGrid, max(Lgrid,1e-300), 'r--', 'LineWidth',1.2);
semilogy(zGrid, max(hL,1e-300), 'k-', 'LineWidth',1.5);
yline(1,':','Color',[.5 .5 .5]); grid on;
legend('h(z)=payoff(z)','L(z)=\phi/\phi_\mu','h(z)L(z)','Location','best');
xlabel('z'); ylabel('log-scale');
title(sprintf('Diagnostics at \\mu=%.3f (goal: flatten h(z)L(z))',muShift));

subplot(1,2,2);
plot(zGrid, phi, 'b-', 'LineWidth',1.2); hold on;
plot(zGrid, phiMu, 'r--', 'LineWidth',1.2);
grid on; legend('\phi(z)','\phi_\mu(z)','Location','best');
xlabel('z'); ylabel('pdf');
title('Original vs proposal density');

%% Question 3: Arithmetic Basket Call by MC

clear; clc;
n     = 12;
S0    = 100 * ones(n,1);       
r     = 0.02;
sigma = 0.4 * ones(n,1);       
rho   = 0.6;                  
T     = 4.0;
disc  = exp(-r*T);

Klist = [80 100 120];
Nlist = [1e3 1e4 1e5];


R = (1-rho)*eye(n) + rho*ones(n);             %  Correlation matrix of Brownian motions
Dsig = diag(sigma);
Sigma = Dsig * R * Dsig * T;                    %Covariance matrix of the log returns over time T

mu = log(S0) + (r - 0.5*sigma.^2)*T;

% Cholesky factor for correlated normals
L = chol(Sigma, 'lower');

fprintf('Arithmetic basket call (equal weights 1/n)\n');
fprintf('n = %d, r = %.2f%%, sigma = %.2f, rho = %.2f, T = %.1f years\n\n',n, 100*r, sigma(1), rho, T);

for K = Klist
    fprintf('Strike K = %d\n', K);
    fprintf('%12s %14s %14s %14s\n', 'N', 'Price', 'SD (payoff)', 'SE (mean)');
    for N = Nlist
        % simulate log S(T)
        Z = randn(n, N);               
        X = mu + L * Z;              
        ST = exp(X);                 
        % arithmetic average per path
        A = mean(ST, 1);                
        Y = disc * max(A - K, 0);
        price = mean(Y);
        sd    = std(Y, 1);              
        se    = sd / sqrt(N);            

        fprintf('%12.0f %14.6f %14.6f %14.6f\n', N, price, sd, se);
    end
    fprintf('\n');
end





%% Question4: Heston model MC vs Fourier 
clear; clc;


S0 = 90;
K  = 80;    % Same parameters as in question 1
r  = 0.01;
T  = 1.0;
disc = exp(-r*T);

% Heston parameters 
V0      = 0.16;   
kappa   = 10;     
theta   = 0.16;   
sigma_v = 0.1;   
rho     = -0.8;   

% MC controls
Npaths = 2e5;     % number of paths
Nsteps = 1000;    % time steps (small dt recommended for Heston)

fprintf('Heston MC (crude) for European CALL\n');
fprintf('S0=%.2f, K=%.2f, r=%.2f%%, T=%.2f\n', S0, K, 100*r, T);
fprintf('V0=%.2f, kappa=%.2f, theta=%.2f, sigma=%.2f, rho=%.2f\n',V0, kappa, theta, sigma_v, rho);

tic;
[mcPrice, mcSD, mcSE] = heston_mc_eur_call_crude( ...
    S0, K, r, T, Npaths, Nsteps, kappa, theta, sigma_v, rho, V0);
tMC = toc;

fprintf('MC price   = %.6f   (sd payoff = %.6f, se mean = %.6f)  [%.2fs]\n',mcPrice, mcSD, mcSE, tMC);

%% Fourier inv

if exist('opt_price','file') == 6
    try
        par   = [V0, kappa, theta, sigma_v, rho];  % <-- CORRECT ORDER
        model = 'Heston';
        call  = 1;     % 1 = call, 0 = put
        power = 1;     % 1 = vanilla payoff (max(S_T-K,0))

        Kvec  = K(:)';           % row vector
        Svec  = S0(:)';          % row vector
        rvec  = r(:)';           % row vector
        Tau   = T(:)';           % row vector

        fftPrice = opt_price(model, par, call, power, Kvec, Svec, rvec, Tau);
        fftPrice = fftPrice(1);   % scalar

        fprintf('Fourier price via opt_price(''Heston'') = %.6f\n', fftPrice);
        fprintf('Abs. diff (MC - Fourier) = %.6f\n\n', abs(mcPrice - fftPrice));

    catch ME
        fprintf(2,'opt_price call failed: %s\n\n', ME.message);
    end
else
    fprintf('(opt_price.p not found on the MATLAB path; skipping Fourier comparison.)\n\n');
end

%% Up-and-out call under BS 

S0_bar = 50;
K_bar  = 40;
B      = 100;        % barrier (up-and-out)
sigma  = 0.6;        % choose a BS vol for this task
T_bar  = 1.0;
r_bar  = 0.01;

Npaths2 = 2e5;
Nsteps2 = 1000;

fprintf('BS up-and-out CALL (crude MC): S0=%.2f, K=%.2f, B=%.2f, r=%.2f%%, T=%.2f, sigma=%.2f\n', ...
        S0_bar, K_bar, B, 100*r_bar, T_bar, sigma);

tic;
[barPrice, barSD, barSE] = bs_up_and_out_call_mc( ...
    S0_bar, K_bar, B, r_bar, sigma, T_bar, Npaths2, Nsteps2);
tBAR = toc;

fprintf('Barrier MC price = %.6f   (sd payoff = %.6f, se mean = %.6f)  [%.2fs]\n', ...
        barPrice, barSD, barSE, tBAR);

%% Home assignment 
%% B.3 – Arithmetic basket call, crude MC + control variate (geometric)
clear; clc;

% ----- Problem data -----
n     = 12;
S0    = 100*ones(n,1);
r     = 0.02;
T     = 4.0;
sigma = 0.4*ones(n,1);
c     = (1/n)*ones(n,1);           % equal weights
Klist = [80 100 120];
Nlist = [1e3 1e4 1e5];

rho   = 0.6;
R     = (1-rho)*eye(n) + rho*ones(n);
Lcorr = chol(R,'lower');


mu = log(S0) + (r - 0.5*sigma.^2)*T;                  
SigRoot = sqrt(T)*diag(sigma)*Lcorr;                    

disc = exp(-r*T);
Phi  = @(x) 0.5*erfc(-x/sqrt(2));

% Black-Scholes call (scalar)
bs_call = @(S,K,vol,r,T) S*Phi((log(S/K)+(r+0.5*vol^2)*T)/(vol*sqrt(T))) - K*exp(-r*T)*Phi((log(S/K)+(r-0.5*vol^2)*T)/(vol*sqrt(T)));

% Geometric basket call price (analytic lower bound)
function CG = geom_basket_call_price(S0, sigma, R, c, r, T, K)
    Phi  = @(x) 0.5*erfc(-x/sqrt(2));
    % m and v for Y = sum c_i log S_i(T) ~ N(m,v)
    m = sum(c .* ( log(S0) + (r - 0.5*sigma.^2)*T ));
    v = T * (c'.*(sigma')) * R * (sigma.*c);   
    if v < 1e-14, v = 1e-14; end              
    d1 = (m - log(K) + v)/sqrt(v);
    d2 = d1 - sqrt(v);
    CG = exp(-r*T) * ( exp(m+0.5*v)*Phi(d1) - K*Phi(d2) );
end

% Analytic lower & upper bounds (vectors over Klist)
Lower = zeros(size(Klist));
Upper = zeros(size(Klist));
for k = 1:numel(Klist)
    K = Klist(k);
    Lower(k) = geom_basket_call_price(S0, sigma, R, c, r, T, K);  % geometric basket call
    Upper(k) = bs_call(S0(1), K, sigma(1), r, T);
end

% MC with control variate 
fprintf('Arithmetic basket call (n=12), r=%.2f%%, vol=0.4, rho=0.6, T=%.1f\n\n',100*r,T);
hdr = '%8s %6s   %12s %10s    %12s %10s    %12s %12s\n';
fmt = '%8s %6.0f   %12.6f %10.6f    %12.6f %10.6f    %12.6f %12.6f\n';
fprintf(hdr,'K','N','Crude','SE(crude)','CV price','SE(CV)','Lower','Upper');

for k = 1:numel(Klist)
    K = Klist(k);

    % analytic mean of the control (geometric basket call)
    Cg = geom_basket_call_price(S0, sigma, R, c, r, T, K);

    for N = Nlist
        Z  = randn(n,N);
        X  = mu + SigRoot*Z;           
        ST = exp(X);

       
        A  = mean(ST,1);               % 1 x N
        Ya = disc * max(A - K, 0);     % crude samples

        % geometric basket payoff (same paths)
        G  = exp( c' * log(ST) );      % row: sum c_i log S_i(T) then exp
        Yg = disc * max(G - K, 0);

        
        crude_price = mean(Ya);
        se_crude    = std(Ya,1)/sqrt(N);
        Yg_c  = Yg - mean(Yg);         
        Ya_c  = Ya - mean(Ya);
        beta  = (Yg_c*Ya_c')/(Yg_c*Yg_c');   
        Zcv   = Ya - beta*(Yg - Cg);         
        cv_price = mean(Zcv);
        se_cv    = std(Zcv,1)/sqrt(N);

        fprintf(fmt, sprintf('%d',K), N, crude_price, se_crude, cv_price, se_cv, Lower(k), Upper(k));
    end
    fprintf('\n');
end



%% Q4(a): Equity-linked note ,find pr by Monte Carlo
clear; clc;

S0 = [141.3; 80.9; 193.3; 316.7; 212.9; 75.6; 236.6; 108.8; 20.68; 192.9; 46.55; 88.95];
c  = [0.07; 0.11; 0.11; 0.18; 0.06; 0.04; 0.01; 0.17; 0.04; 0.02; 0.03; 0.16];   % sum(c)=1
sigma = [0.32; 0.22; 0.31; 0.18; 0.27; 0.22; 0.21; 0.23; 0.34; 0.17; 0.18; 0.23];
r = -0.001;                 
T = 5.0;                 
NA = 100;                   
targetPrice = 1.1*NA;       

% Correlation matrix (12×12).
R = [ ...
 1.00 0.30 0.28 0.32 0.36 0.52 0.29 0.37 0.44 0.41 0.38 0.41
 0.30 1.00 0.17 0.45 0.44 0.42 0.37 0.35 0.32 0.42 0.42 0.27
 0.28 0.17 1.00 0.20 0.17 0.31 0.23 0.24 0.20 0.25 0.27 0.21
 0.32 0.45 0.20 1.00 0.34 0.38 0.39 0.30 0.32 0.43 0.42 0.30
 0.36 0.44 0.17 0.34 1.00 0.47 0.22 0.33 0.39 0.38 0.31 0.31
 0.52 0.42 0.31 0.38 0.47 1.00 0.37 0.37 0.46 0.47 0.42 0.56
 0.29 0.37 0.23 0.39 0.22 0.37 1.00 0.25 0.26 0.42 0.29 0.22
 0.37 0.35 0.24 0.30 0.33 0.37 0.25 1.00 0.39 0.39 0.34 0.26
 0.44 0.32 0.20 0.32 0.39 0.46 0.26 0.39 1.00 0.43 0.38 0.29
 0.41 0.42 0.25 0.43 0.38 0.47 0.42 0.39 0.43 1.00 0.46 0.38
 0.38 0.42 0.27 0.42 0.31 0.42 0.29 0.34 0.38 0.46 1.00 0.39
 0.41 0.27 0.21 0.30 0.31 0.56 0.22 0.26 0.29 0.38 0.39 1.00 ];


%Monte Carlo settings
N  = 2e6;          % number of paths (increase/decrease as needed)
use_control_variate = true;   % turn on/off geometric control variate

%Estimate C0 = E[(sum c_i * S_i(T)/S_i(0) - 1)^+]
tic;
if ~use_control_variate
    [C0, seC0, ciC0] = mc_C0_crude(c, sigma, R, r, T, N);
    tag = 'Crude MC';
else
    [C0, seC0, ciC0, rhoCV] = mc_C0_with_geomCV(c, sigma, R, r, T, N);
    tag = sprintf('MC + Geometric CV (rho=%.3f)', rhoCV);
end
tMC = toc;

%Solve for pr such that V0(pr) = 1.1*NA
disc = exp(-r*T);
pr  = (targetPrice/NA - disc) / (disc*C0);

% Propagate SE: pr = const / C0  =>  Var(pr) ≈ (const/C0^2)^2 Var(C0)
const = (targetPrice/NA - disc) / disc;
se_pr = abs(const) * seC0 / (C0^2);
z = 1.96;
ci_pr = [pr - z*se_pr, pr + z*se_pr];
fprintf('\n=== Equity-linked note – participation ratio pr ===\n');
fprintf('Estimator: %s\n', tag);
fprintf('Paths N: %g, Time: %.2fs\n', N, tMC);
fprintf('C0 (undiscounted expectation) = %.6f  (SE = %.6f), 95%% CI [%.6f, %.6f]\n', C0, seC0, ciC0(1), ciC0(2));
fprintf('disc = e^{-rT} = %.8f\n', disc);
fprintf('pr = %.6f  (SE = %.6f), 95%% CI [%.6f, %.6f]\n\n', pr, se_pr, ci_pr(1), ci_pr(2));

%% Part (b): price at t=1 

S1_now = [179.3, 57.45, 163.7, 260.6, 220.4, 89.85, 259.5, 143.8, 20.83, 194.7, 37.62, 97.5]';  
Delta   = 4.0;                     
disc1   = exp(-r*Delta);           


Dsig  = diag(sigma);
Sigma = (Dsig*R*Dsig) * Delta;    
L     = chol(Sigma,'lower');       
m  = sum(c .* (log(S1_now) - log(S0))) + Delta*sum(c.*(r - 0.5*sigma.^2));
v  = Delta * ( (c.*sigma)' * R * (c.*sigma) );
sqrtv = sqrt(v);
d1 = (m + v)/sqrtv;
d2 = d1 - sqrtv;
EY = exp(m + 0.5*v)*normcdf(d1) - normcdf(d2);   

N  = 2e6;                         
Z  = randn(12,N);
Xn = (log(S1_now) + (r - 0.5*sigma.^2)*Delta) + L*Z;  
ST = exp(Xn);                                      

A  = (c' * (ST ./ S0))';          
X  = max(A - 1, 0);               

% Control variate sample:
Y  = exp( sum((c.*(Xn - log(S0))) ,1)' ) - 1;   % G_T - 1  (undiscounted)

% Optimal beta
covXY = cov(X,Y);                  % 2x2 matrix; covXY(1,2) is covariance
beta  = covXY(1,2)/covXY(2,2);
Xbar  = mean(X);
Ybar  = mean(Y);
C1_CV = Xbar - beta*(Ybar - EY);   % unbiased CV estimator of C1
se_C1 = std(X - beta*(Y - EY),1)/sqrt(N);
V1 = NA*disc1 + NA*pr*disc1*C1_CV;
se_V1 = NA*pr*disc1*se_C1;

fprintf('\n=== Part (b): Price at t=1 given S(1) ===\n');
fprintf('N = %.0f, Delta = %.1f years\n', N, Delta);
fprintf('disc1 = e^{-r*Delta} = %.8f\n', disc1);
fprintf('E[(G_T-1)^+|F1] = %.6f (analytic CV)\n', EY);
fprintf('C1 (undiscounted)  = %.6f  (SE = %.6f)\n', C1_CV, se_C1);
fprintf('V1 = NA*disc + NA*pr*disc*C1 = %.6f  (SE = %.6f)\n', V1, se_V1);
fprintf('95%% CI for V1: [%.6f, %.6f]\n', V1 - 1.96*se_V1, V1 + 1.96*se_V1);

%% Functions

function [C0, se, ci] = mc_C0_crude(c, sigma, R, r, T, N)
% Monte Carlo estimate of C0 (undiscounted) using crude MC
    n = numel(c);
    L = chol(R,'lower');
    Z = L * randn(n,N);                    % correlated normals
    drift   = (r - 0.5*sigma.^2)*T;        % n×1
    volterm = sigma*sqrt(T);               % n×1
    G = exp(drift + volterm .* Z);         % growth factors S_i(T)/S_i(0)
    B = c.' * G;                           % basket of growth factors
    Y = max(B - 1, 0);                     % payoff (no discount)
    C0 = mean(Y);
    sd = std(Y,1); se = sd/sqrt(N);
    z=1.96; ci = [C0 - z*se, C0 + z*se];
end

function [C0_CV, seCV, ciCV, rhoCV] = mc_C0_with_geomCV(c, sigma, R, r, T, N)
% Control variate with geometric basket (closed-form expectation known)
    n = numel(c);
    L = chol(R,'lower');
    Z = L * randn(n,N);

    drift   = (r - 0.5*sigma.^2)*T;        % n×1
    volterm = sigma*sqrt(T);               % n×1
    G = exp(drift + volterm .* Z);         % growth factors
    B = c.' * G;                           % arithmetic (of growth factors)
    Y = max(B - 1, 0);                     % target

    % Geometric basket pathwise
    Ylog = sum( c .* (drift + volterm .* Z), 1 );   % 1×N
    Ggeo = exp(Ylog);
    Zgeo = max(Ggeo - 1, 0);              % control variate

    % Closed-form E[Zgeo] (UNDISCOUNTED)
    m = sum( c .* (r - 0.5*sigma.^2) ) * T;
    v = T * sum( (c.*sigma) .* (R * (c.*sigma)) );   % c⊙σ * R * c⊙σ
    if v<=0, error('Variance v must be positive.'); end
    d1 = (m + v) / sqrt(v);    % since K=1 => ln K = 0
    d2 = d1 - sqrt(v);
    Phi = @(x) 0.5*(1+erf(x/sqrt(2)));
    EZgeo = exp(m + 0.5*v) * Phi(d1) - 1 * Phi(d2);

    % Optimal b (sample)
    covYZ = cov(Y(:), Zgeo(:));    % 2×2
    bstar = covYZ(1,2) / covYZ(2,2);
    rhoCV = covYZ(1,2) / sqrt(covYZ(1,1)*covYZ(2,2));

    % CV estimator
    Ycv = Y - bstar*(Zgeo - EZgeo);
    C0_CV = mean(Ycv);
    sdCV = std(Ycv,1);  seCV = sdCV/sqrt(N);
    z=1.96; ciCV = [C0_CV - z*seCV, C0_CV + z*seCV];
end

function [price, sdY, seY] = heston_mc_eur_call_crude( ...
        S0, K, r, T, Npaths, Nsteps, kappa, theta, sigma_v, rho, V0)
% Crude Monte Carlo for a European call under the Heston model
% Milstein on V with full truncation; log-Euler on S.

    dt   = T/Nsteps;
    sqdt = sqrt(dt);

    S = S0 * ones(Npaths,1);
    V = V0 * ones(Npaths,1);

    for n = 1:Nsteps
        % Two independent N(0,1)
        Z2 = randn(Npaths,1);
        Z1 = randn(Npaths,1);
        dW2 = sqdt * Z2;
        dW1 = sqdt * (rho.*Z2 + sqrt(1-rho^2).*Z1);

        % Milstein step 

        % Euler/log step for S under risk-neutral drift r 
        S = S .* exp( (r - 0.5*V)*dt + sqrt(V).*dW1 );
        Vpos = max(V,0);
        V    = V + kappa*(theta - Vpos)*dt+ sigma_v.*sqrt(Vpos).*dW2 + 0.25*sigma_v^2*((dW2).^2 - dt);
        V    = max(V,0);  % enforce non-negativity

    end

    payoff = max(S - K, 0);
    Y      = exp(-r*T) * payoff;

    price  = mean(Y);
    sdY    = std(Y,1);          
    seY    = sdY / sqrt(Npaths);
end

function [price, sdY, seY] = bs_up_and_out_call_mc( ...
        S0, K, B, r, sigma, T, Npaths, Nsteps)
% Crude MC for an up-and-out call under geometric BM.
% Knocked out if the path ever hits/exceeds B at monitoring times.

    dt   = T/Nsteps;
    sqdt = sqrt(dt);

    S = S0 * ones(Npaths,1);
    alive = true(Npaths,1);

    for n = 1:Nsteps
        Z = randn(Npaths,1);
        % Evolve only paths which are still alive (not knocked out)
        idx = alive;
        S(idx) = S(idx) .* exp((r-0.5*sigma^2)*dt + sigma*sqdt.*Z(idx));

        % Check barrier crossing on the grid
        crossed = S >= B;
        alive(crossed) = false;
    end

    payoff = zeros(Npaths,1);
    payoff(alive) = max(S(alive) - K, 0);

    Y     = exp(-r*T) * payoff;
    price = mean(Y);
    sdY   = std(Y,1);
    seY   = sdY / sqrt(Npaths);
end