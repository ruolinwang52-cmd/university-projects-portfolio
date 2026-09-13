%% Section 2.1 Load option data (OMXS30optA) and save some hold out obs
clear; clc; close all;
load('OMXS30optA.mat');   
T = numel(opt);           
holdFrac = 0.20;          % 20% hold out data
rng(1);                   % reproducible split
isHold = cell(T,1);
idxHold = cell(T,1);
idxCal  = cell(T,1);

for t = 1:T
    % Use MID data
    P = opt(t).MID(:);
    K = opt(t).K(:);
    Tau = opt(t).T(:);        % time-to-maturity
    C = opt(t).C(:);          % 1=call, 0=put
    S = opt(t).S(:);
    r = opt(t).r(:);
    good = isfinite(P) & isfinite(K) & isfinite(Tau) & isfinite(S) & isfinite(r) & (P > 0) & (K > 0) & (Tau > 0) & (S > 0);
    I = find(good);
    N = numel(I);
    if N < 10
        isHold{t} = false(numel(P),1);
        idxHold{t} = [];
        idxCal{t}  = I;
        continue
    end

    m = K(I) ./ S(I);
    %  80/20 split across (maturity x moneyness) bins 
    % moneyness proxy: K/S
    % Define bin edges 
    mEdges   = [0, 0.9, 0.97, 1.03, 1.10, inf];        % deep ITM ... deep OTM
    tauEdges = quantile(Tau(I), [0, 0.33, 0.66, 1]);    % 3 maturity buckets
    tauEdges(end) = tauEdges(end) + 1e-12;             % include max
    holdMaskLocal = false(N,1);
    for a = 1:(numel(mEdges)-1)
        for b = 1:(numel(tauEdges)-1)
            inBin = (m >= mEdges(a) & m < mEdges(a+1)) & (Tau(I) >= tauEdges(b) & Tau(I) < tauEdges(b+1));
            binIdx = find(inBin);
            nb = numel(binIdx);
            if nb == 0, continue; end

            nh = max(1, round(holdFrac * nb));  % at least 1 per non-empty bin
            perm = binIdx(randperm(nb));
            holdMaskLocal(perm(1:nh)) = true;
        end
    end

    % Map back to full-length mask
    isHold_t = false(numel(P),1);
    isHold_t(I(holdMaskLocal)) = true;

    isHold{t} = isHold_t;
    idxHold{t} = find(isHold_t);
    idxCal{t}  = find(good & ~isHold_t);
end
%% Section 2.2 UKF on BS: time-varying volatility sigma_t (state = log(sigma_t))

% UKF tuning 
alpha = 1e-3;    
beta  = 2;       
kappa = 0;
n = 1;                       % x_t = log(sigma_t), so sigma always positive
xHat = zeros(T,1);           % filtered state mean
PHat = zeros(T,1);           % filtered state variance
% Process/measurement noise tuning
Q = 1e-5;                    % process variance of log-sigma increments
R = (0.5)^2;                 % price noise variance 
% Initial state
sigma0 = 0.20;               % initial vol guess 
x = log(sigma0);
Pxx = 0.2^2;                 % prior variance of log-sigma

for t = 1:T

    xPred = x;               %Prediction
    PPred = Pxx + Q;   % \mathrm{Var}(x_t \mid \mathcal F_{t-1})


    % Update step using all calibration options at time t
    calIdx = idxCal{t};
    if isempty(calIdx)
        x = xPred; Pxx = PPred;
        xHat(t) = x; PHat(t) = Pxx;
        continue
    end

    %  updates
    for j = 1:numel(calIdx)
        i = calIdx(j);

        yObs = opt(t).MID(i);
        Ci   = opt(t).C(i);          % 1 call, 0 put
        Ki   = opt(t).K(i);
        Si   = opt(t).S(i);
        ri   = opt(t).r(i);
        Taui = opt(t).T(i);
        powi = 1;                    % plain vanilla payoff power

        hfun = @(xstate) opt_price('BS', exp(xstate), Ci, powi, Ki, Si, ri, Taui);
        [xPred, PPred] = ukf_update_scalar(xPred, PPred, yObs, hfun, R, alpha, beta, kappa);
    end

    % Save filtered
    x = xPred; Pxx = PPred;
    xHat(t) = x;
    PHat(t) = Pxx;
end

sigmaHat = exp(xHat);
figure;
plot(1:T, sigmaHat, 'LineWidth', 1.2);
xlabel('Time index (10-min snapshot)');
ylabel('Filtered BS volatility \sigma_t');
title('UKF filtered Black–Scholes volatility');
grid on;

%% Section 2.3 Evaluate pricing on hold out set (interpolation + time extrapolation)

rmse_in = NaN(T,1);
mae_in  = NaN(T,1);
rmse_1s = NaN(T-1,1);   
mae_1s  = NaN(T-1,1);

for t = 1:T
    sig_t = sigmaHat(t);

    holdIdx = idxHold{t};
    if isempty(holdIdx)
        continue
    end

    % Interpolation at same time t
    yObs = opt(t).MID(holdIdx);
    yHat = zeros(numel(holdIdx),1);

    for j = 1:numel(holdIdx)
        i = holdIdx(j);
        yHat(j) = opt_price('BS', sig_t, opt(t).C(i), 1, opt(t).K(i), opt(t).S(i), opt(t).r(i), opt(t).T(i));
    end

    err = yObs - yHat;
    rmse_in(t) = sqrt(mean(err.^2));
    mae_in(t)  = mean(abs(err));

    %Time extrapolation: use sigma_t to price at t+1 
    if t < T
        holdIdx_next = idxHold{t+1};
        if ~isempty(holdIdx_next)

            yObs2 = opt(t+1).MID(holdIdx_next);
            yHat2 = zeros(numel(holdIdx_next),1);

            for j = 1:numel(holdIdx_next)
                i = holdIdx_next(j);
                yHat2(j) = opt_price('BS', sig_t, opt(t+1).C(i), 1, opt(t+1).K(i), opt(t+1).S(i), opt(t+1).r(i), opt(t+1).T(i));
            end

            err2 = yObs2 - yHat2;
            rmse_1s(t) = sqrt(mean(err2.^2));
            mae_1s(t)  = mean(abs(err2));
        end
    end
end

% Plot error summaries
figure;
plot(rmse_in, 'LineWidth', 1.2); hold on;
plot([rmse_1s; NaN], 'LineWidth', 1.2);  % align lengths
xlabel('Time index (10-min snapshot)');
ylabel('RMSE on hold-out options');
title('Hold-out RMSE: interpolation (same t) vs time extrapolation (t -> t+1)');
legend('Interpolation (same t)', 'Time extrapolation (t \rightarrow t+1)', 'Location', 'best');
grid on;
fprintf('\nAverage hold-out RMSE (interpolation): %.6f\n', mean(rmse_in,'omitnan'));
fprintf('Average hold-out RMSE (time extrapolation): %.6f\n\n', mean(rmse_1s,'omitnan'));

%% Section 2.4 Hold-out pricing errors by moneyness & maturity (quantile bins, BS + UKF)

% Fixed moneyness bins, same as Section 2.1
mEdges  = [0, 0.9, 0.97, 1.03, 1.10, inf];
mLabels = {'Deep ITM','ITM','ATM','OTM','Deep OTM'};
nM = numel(mLabels);
nT = 3;   % Short / Medium / Long (quantile-based)
SSE   = zeros(nM, nT);   % Sum of squared errors
COUNT = zeros(nM, nT);   % Number of hold-out options

% Loop over time snapshots
for t = 1:T

    holdIdx = idxHold{t};
    if isempty(holdIdx)
        continue
    end

    sig_t = sigmaHat(t);

    % Extract hold-out data
    Pobs = opt(t).MID(holdIdx);
    K    = opt(t).K(holdIdx);
    S    = opt(t).S(holdIdx);
    Tau  = opt(t).T(holdIdx);   % maturity in years
    C    = opt(t).C(holdIdx);
    r_t  = opt(t).r(holdIdx);
    % Moneyness
    m = K ./ S;
    %same logic as Section 2.1
    tauEdges = quantile(Tau, [0, 0.33, 0.66, 1]);
    tauEdges(end) = tauEdges(end) + 1e-12;

    % Model prices
    Phat = zeros(size(Pobs));
    for j = 1:numel(holdIdx)
        Phat(j) = opt_price('BS', sig_t, C(j), 1, K(j), S(j), r_t(j), Tau(j));
    end

    err = Pobs - Phat;

    % Bin errors
    for a = 1:nM
        inM = (m >= mEdges(a)) & (m < mEdges(a+1));
        for b = 1:nT
            inT = (Tau >= tauEdges(b)) & (Tau < tauEdges(b+1));
            inBin = inM & inT;

            nb = sum(inBin);
            if nb > 0
                SSE(a,b)   = SSE(a,b)   + sum(err(inBin).^2);
                COUNT(a,b) = COUNT(a,b) + nb;
            end
        end
    end
end

% Compute RMSE per bin
RMSE = NaN(nM, nT);
for a = 1:nM
    for b = 1:nT
        if COUNT(a,b) > 0
            RMSE(a,b) = sqrt(SSE(a,b) / COUNT(a,b));
        end
    end
end
tauLabels = {'Short (Q1)','Medium (Q2)','Long (Q3)'};
RMSE_table = array2table(RMSE, 'VariableNames', tauLabels, 'RowNames', mLabels);
COUNT_table = array2table(COUNT,  'VariableNames', tauLabels,  'RowNames', mLabels);
disp('Hold-out RMSE by moneyness and maturity (quantile-based, BS + UKF)');
disp(RMSE_table);
disp('Counts per bin');
disp(COUNT_table);

%% Section 2.5 UKF on Heston: calibrate par_t = [v0, kappa, theta, sigma, rho]
% State in transformed space (enforce constraints):
% x = [log(v0); log(kappa); log(theta); log(sigma); atanh(rho)]
% so:
%   v0,kappa,theta,sigma > 0 and rho in (-1,1)

alphaUKF = 1e-3; betaUKF = 2; kappaUKF = 0;

nH = 5;
xHatH = zeros(nH, T);
PHatH = zeros(nH, nH, T);

% initial guess 
v0_0    = 0.04;   % (20% vol)^2
kappa0  = 2.0;
theta0  = 0.04;   % long-run variance
sigma0  = 0.5;    % vol-of-vol
rho0    = -0.5;

x  = [log(v0_0); log(kappa0); log(theta0); log(sigma0); atanh(rho0)];

% Prior covariance + random-walk process noise, can be tunned
Pxx = diag([0.5, 0.5, 0.5, 0.5, 0.5].^2);
QH  = diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-5]);   % rho usually moves less
RH  = (0.5)^2;                                  % option price noise variance (tune)

for t = 1:T

    % Prediction (random walk)
    xPred = x;
    PPred = Pxx + QH;

    calIdx = idxCal{t};
    if isempty(calIdx)
        x = xPred; Pxx = PPred;
        xHatH(:,t) = x; PHatH(:,:,t) = Pxx;
        continue
    end

    % Sequential updates using calibration options at time t
    for j = 1:numel(calIdx)
        i = calIdx(j);

        yObs = opt(t).MID(i);

        Ci   = opt(t).C(i);
        Ki   = opt(t).K(i);
        Si   = opt(t).S(i);
        ri   = opt(t).r(i);
        Taui = opt(t).T(i);
        powi = 1;

        hfun = @(xstate) heston_price_from_state(xstate, Ci, powi, Ki, Si, ri, Taui);

        [xPred, PPred] = ukf_update_scalar(xPred, PPred, yObs, hfun, RH, alphaUKF, betaUKF, kappaUKF);
    end

    x = xPred; Pxx = PPred;
    xHatH(:,t) = x;
    PHatH(:,:,t) = Pxx;
end

% Extract parameter paths (each is T x 1)
v0Hat    = exp(xHatH(1,:)).';
kappaHat = exp(xHatH(2,:)).';
thetaHat = exp(xHatH(3,:)).';
sigmaHatH= exp(xHatH(4,:)).';
rhoHat   = tanh(xHatH(5,:)).';
% Plot parameter paths
figure; plot(1:T, sqrt(v0Hat), 'LineWidth', 1.2);
xlabel('Time index (10-min snapshot)'); ylabel('\surd v_{0,t}');
title('Heston UKF: sqrt(v0_t)'); grid on;
figure; hold on;
plot(1:T, kappaHat, 'LineWidth', 1.2);
plot(1:T, thetaHat, 'LineWidth', 1.2);
plot(1:T, sigmaHatH,'LineWidth', 1.2);
plot(1:T, rhoHat,   'LineWidth', 1.2);
xlabel('Time index (10-min snapshot)'); ylabel('Value');
title('Heston UKF: parameters'); legend('\kappa','\theta','\sigma','\rho','Location','best');
grid on;

%% Section 2.6 Evaluate Heston UKF pricing on hold-out set (interpolation + time extrapolation)

rmseH_in = NaN(T,1);
maeH_in  = NaN(T,1);
rmseH_1s = NaN(T-1,1);
maeH_1s  = NaN(T-1,1);

for t = 1:T

    holdIdx = idxHold{t};
    if isempty(holdIdx), continue; end

    % Build par_t in the EXACT opt_price order: [v0, kappa, theta, sigma, rho]
    par_t = [v0Hat(t), kappaHat(t), thetaHat(t), sigmaHatH(t), rhoHat(t)];
    yObs = opt(t).MID(holdIdx);%Interpolation at time t 
    yHat = zeros(numel(holdIdx),1);
    for j = 1:numel(holdIdx)
        i = holdIdx(j);
        yHat(j) = opt_price('Heston', par_t, opt(t).C(i), 1, opt(t).K(i), opt(t).S(i), opt(t).r(i), opt(t).T(i));
    end
    err = yObs - yHat;
    rmseH_in(t) = sqrt(mean(err.^2));
    maeH_in(t)  = mean(abs(err));

    % Time extrapolation, use par_t to price hold-out at t+1 
    if t < T
        holdIdx_next = idxHold{t+1};
        if ~isempty(holdIdx_next)
            yObs2 = opt(t+1).MID(holdIdx_next);
            yHat2 = zeros(numel(holdIdx_next),1);
            for j = 1:numel(holdIdx_next)
                i = holdIdx_next(j);
                yHat2(j) = opt_price('Heston', par_t, opt(t+1).C(i), 1, opt(t+1).K(i), opt(t+1).S(i), opt(t+1).r(i), opt(t+1).T(i));
            end
            err2 = yObs2 - yHat2;
            rmseH_1s(t) = sqrt(mean(err2.^2));
            maeH_1s(t)  = mean(abs(err2));
        end
    end
end

figure;
plot(rmseH_in, 'LineWidth', 1.2); hold on;
plot([rmseH_1s; NaN], 'LineWidth', 1.2);
xlabel('Time index (10-min snapshot)'); ylabel('RMSE on hold-out options');
title('Heston+UKF RMSE: interpolation vs time extrapolation');
legend('Interpolation (same t)','Time extrapolation (t \rightarrow t+1)','Location','best');
grid on;
fprintf('\nHeston+UKF Average hold-out RMSE (interpolation): %.6f\n', mean(rmseH_in,'omitnan'));
fprintf('Heston+UKF Average hold-out RMSE (time extrapolation): %.6f\n\n', mean(rmseH_1s,'omitnan'));

%% Section 2.7 Hold-out RMSE by moneyness & maturity (quantile bins, Heston + UKF)

mEdges  = [0, 0.9, 0.97, 1.03, 1.10, inf];
mLabels = {'Deep ITM','ITM','ATM','OTM','Deep OTM'};
nM = numel(mLabels);
nTbin = 3;
SSE_H   = zeros(nM, nTbin);
COUNT_H = zeros(nM, nTbin);

for t = 1:T

    holdIdx = idxHold{t};
    if isempty(holdIdx), continue; end

    par_t = [v0Hat(t), kappaHat(t), thetaHat(t), sigmaHatH(t), rhoHat(t)];

    Pobs = opt(t).MID(holdIdx);
    K    = opt(t).K(holdIdx);
    S    = opt(t).S(holdIdx);
    Tau  = opt(t).T(holdIdx);
    C    = opt(t).C(holdIdx);
    r_t  = opt(t).r(holdIdx);

    m = K ./ S;

    tauEdges = quantile(Tau, [0, 0.33, 0.66, 1]);
    tauEdges(end) = tauEdges(end) + 1e-12;

    Phat = zeros(size(Pobs));
    for j = 1:numel(holdIdx)
        Phat(j) = opt_price('Heston', par_t, C(j), 1, K(j), S(j), r_t(j), Tau(j));
    end

    err = Pobs - Phat;

    for a = 1:nM
        inM = (m >= mEdges(a)) & (m < mEdges(a+1));
        for b = 1:nTbin
            inT = (Tau >= tauEdges(b)) & (Tau < tauEdges(b+1));
            inBin = inM & inT;
            nb = sum(inBin);
            if nb > 0
                SSE_H(a,b)   = SSE_H(a,b)   + sum(err(inBin).^2);
                COUNT_H(a,b) = COUNT_H(a,b) + nb;
            end
        end
    end
end

RMSE_H = NaN(nM, nTbin);
for a = 1:nM
    for b = 1:nTbin
        if COUNT_H(a,b) > 0
            RMSE_H(a,b) = sqrt(SSE_H(a,b) / COUNT_H(a,b));
        end
    end
end

tauLabelsQ = {'Short (Q1)','Medium (Q2)','Long (Q3)'};
RMSEH_table  = array2table(RMSE_H,  'VariableNames', tauLabelsQ, 'RowNames', mLabels);
COUNTH_table = array2table(COUNT_H, 'VariableNames', tauLabelsQ, 'RowNames', mLabels);
disp('Hold-out RMSE by moneyness and maturity (quantile-based, Heston + UKF)');
disp(RMSEH_table);
disp('Counts per bin');
disp(COUNTH_table); %Heston plus UKF is a lot better than BS+ UFK

%% Section 2.8 UKF on Bates with FIXED Heston part
% Bates par order for opt_price('Bates'):
% parB = [V0, kappa, theta, sigma, rho, lambda, mu_j, sigma_j]  (1x8 row)
%
% We FIX [V0,kappa,theta,sigma,rho] from Heston UKF outputs,
% and only UKF-filter the jump parameters: [log(lambda); mu_j; log(sigma_j)].

alphaUKF = 1e-3; betaUKF = 2; kappaUKF = 0;

nJ = 3;                              % jump-state dimension
xHatJ = zeros(nJ, T);
PHatJ = zeros(nJ, nJ, T);

%Initial guess for jump params 
lambda0  = 0.8;     % jump intensity
mu_j0    = -0.2;   % mean log jump
sigma_j0 = 0.20;    % jump std (log jump)

x  = [log(lambda0);
      mu_j0;
      log(sigma_j0)];

Pxx = diag([0.2, 0.8, 0.8].^2);   % prior uncertainty    can tune
QJ  = diag([1e-6, 5e-2, 5e-2]);   % jump-parameter drift    can tune
RB  = (0.5)^2;                    % same price noise as before 

for t = 1:T

    % Prediction (random walk)
    xPred = x;
    PPred = Pxx + QJ;

    calIdx = idxCal{t};
    if isempty(calIdx)
        x = xPred; Pxx = PPred;
        xHatJ(:,t) = x; PHatJ(:,:,t) = Pxx;
        continue
    end

    % FIXED Heston part at time t (from Heston UKF)
    V0_t    = v0Hat(t);
    kappa_t = kappaHat(t);
    theta_t = thetaHat(t);
    sigma_t = sigmaHatH(t);
    rho_t   = rhoHat(t);

    % Sequential updates using calibration options at time t
    for j = 1:numel(calIdx)
        i = calIdx(j);
        yObs = opt(t).MID(i);
        Ci   = opt(t).C(i);
        Ki   = opt(t).K(i);
        Si   = opt(t).S(i);
        ri   = opt(t).r(i);
        Taui = opt(t).T(i);
        powi = 1;

        hfun = @(xstate) bates_price_from_jumpstate(xstate, V0_t, kappa_t, theta_t, sigma_t, rho_t, Ci, powi, Ki, Si, ri, Taui);

        [xPred, PPred] = ukf_update_scalar(xPred, PPred, yObs, hfun, RB, alphaUKF, betaUKF, kappaUKF);
    end

    x = xPred; Pxx = PPred;
    xHatJ(:,t) = x;
    PHatJ(:,:,t) = Pxx;
end

% Extract jump parameter paths
lambdaHat  = exp(xHatJ(1,:)).';
mu_jHat    = xHatJ(2,:).';
sigma_jHat = exp(xHatJ(3,:)).';

% Plot jump params
figure;
plot(1:T, lambdaHat, 'LineWidth', 1.2); hold on;
plot(1:T, mu_jHat, 'LineWidth', 1.2);
plot(1:T, sigma_jHat, 'LineWidth', 1.2);
xlabel('Time index (10-min snapshot)');
ylabel('Value');
title('Bates (fixed Heston) UKF: jump parameters');
legend('\lambda','\mu_j','\sigma_j','Location','best');
grid on;


%% Section 2.9 Hold-out pricing errors (Bates with fixed Heston + UKF jump params)

rmseB_in = NaN(T,1);
maeB_in  = NaN(T,1);
rmseB_1s = NaN(T-1,1);
maeB_1s  = NaN(T-1,1);

for t = 1:T

    holdIdx = idxHold{t};
    if isempty(holdIdx), continue; end

    % Fixed Heston at time t
    V0_t    = v0Hat(t);
    kappa_t = kappaHat(t);
    theta_t = thetaHat(t);
    sigma_t = sigmaHatH(t);
    rho_t   = rhoHat(t);

    % Jump params at time t
    lam_t  = lambdaHat(t);
    muj_t  = mu_jHat(t);
    sigj_t = sigma_jHat(t);

    parB_t = [V0_t, kappa_t, theta_t, sigma_t, rho_t, lam_t, muj_t, sigj_t];  % 1x8 row

    % Interpolation (same t)
    yObs = opt(t).MID(holdIdx);
    yHat = zeros(numel(holdIdx),1);
    for j = 1:numel(holdIdx)
        i = holdIdx(j);
        yHat(j) = opt_price('Bates', parB_t, opt(t).C(i), 1, opt(t).K(i), opt(t).S(i), opt(t).r(i), opt(t).T(i));
    end
    err = yObs - yHat;
    rmseB_in(t) = sqrt(mean(err.^2));
    maeB_in(t)  = mean(abs(err));

    %Time extrapolation t -> t+1 using parameters from time t
    if t < T
        holdIdx_next = idxHold{t+1};
        if ~isempty(holdIdx_next)
            yObs2 = opt(t+1).MID(holdIdx_next);
            yHat2 = zeros(numel(holdIdx_next),1);
            for j = 1:numel(holdIdx_next)
                i = holdIdx_next(j);
                yHat2(j) = opt_price('Bates', parB_t, opt(t+1).C(i), 1, opt(t+1).K(i), opt(t+1).S(i), opt(t+1).r(i), opt(t+1).T(i));
            end
            err2 = yObs2 - yHat2;
            rmseB_1s(t) = sqrt(mean(err2.^2));
            maeB_1s(t)  = mean(abs(err2));
        end
    end
end

figure;
plot(rmseB_in, 'LineWidth', 1.2); hold on;
plot([rmseB_1s; NaN], 'LineWidth', 1.2);
xlabel('Time index (10-min snapshot)');
ylabel('RMSE on hold-out options');
title('Bates (fixed Heston) RMSE: interpolation vs time extrapolation');
legend('Interpolation (same t)','Time extrapolation (t \rightarrow t+1)','Location','best');
grid on;
fprintf('\nBates(fixed Heston)+UKF Average RMSE (interpolation): %.6f\n', mean(rmseB_in,'omitnan'));
fprintf('Bates(fixed Heston)+UKF Average RMSE (time extrapolation): %.6f\n\n', mean(rmseB_1s,'omitnan'));

%% Section 2.10 RMSE by moneyness & maturity (quantile bins, Bates fixed Heston + UKF)

mEdges  = [0, 0.9, 0.97, 1.03, 1.10, inf];
mLabels = {'Deep ITM','ITM','ATM','OTM','Deep OTM'};
nMny = numel(mLabels);
nTau = 3;

SSE_B   = zeros(nMny, nTau);
COUNT_B = zeros(nMny, nTau);

for t = 1:T

    holdIdx = idxHold{t};
    if isempty(holdIdx), continue; end

    V0_t    = v0Hat(t);
    kappa_t = kappaHat(t);
    theta_t = thetaHat(t);
    sigma_t = sigmaHatH(t);
    rho_t   = rhoHat(t);

    lam_t  = lambdaHat(t);
    muj_t  = mu_jHat(t);
    sigj_t = sigma_jHat(t);

    parB_t = [V0_t, kappa_t, theta_t, sigma_t, rho_t, lam_t, muj_t, sigj_t];

    Pobs = opt(t).MID(holdIdx);
    K    = opt(t).K(holdIdx);
    S    = opt(t).S(holdIdx);
    Tau  = opt(t).T(holdIdx);
    C    = opt(t).C(holdIdx);
    r_t  = opt(t).r(holdIdx);

    m = K ./ S;
    tauEdges = quantile(Tau, [0 0.33 0.66 1]);
    tauEdges(end) = tauEdges(end) + 1e-12;

    Phat = zeros(size(Pobs));
    for j = 1:numel(holdIdx)
        Phat(j) = opt_price('Bates', parB_t, C(j), 1, K(j), S(j), r_t(j), Tau(j));
    end

    err = Pobs - Phat;

    for a = 1:nMny
        inM = (m >= mEdges(a)) & (m < mEdges(a+1));
        for b = 1:nTau
            inT = (Tau >= tauEdges(b)) & (Tau < tauEdges(b+1));
            inBin = inM & inT;

            nb = sum(inBin);
            if nb > 0
                SSE_B(a,b)   = SSE_B(a,b) + sum(err(inBin).^2);
                COUNT_B(a,b) = COUNT_B(a,b) + nb;
            end
        end
    end
end

RMSE_B = NaN(nMny, nTau);
for a = 1:nMny
    for b = 1:nTau
        if COUNT_B(a,b) > 0
            RMSE_B(a,b) = sqrt(SSE_B(a,b) / COUNT_B(a,b));
        end
    end
end

RMSE_B_table = array2table(RMSE_B, 'VariableNames', {'Short(Q1)','Medium(Q2)','Long(Q3)'}, 'RowNames', mLabels);
COUNT_B_table = array2table(COUNT_B,'VariableNames', {'Short(Q1)','Medium(Q2)','Long(Q3)'}, 'RowNames', mLabels);
disp('Bates(fixed Heston)+UKF RMSE by moneyness and maturity (quantile-based)');
disp(RMSE_B_table);
disp('Counts per bin');
disp(COUNT_B_table);


function [xNew, PNew] = ukf_update_scalar(x, P, y, hfun, R, alpha, beta, kappa)


    n = numel(x);
    lambda = alpha^2*(n + kappa) - n;
    c = n + lambda;
    S = chol(c*P, 'lower');% Sigma points
    X = [x, x + S, x - S];          % n x (2n+1)
    % Weights
    Wm = [lambda/c, repmat(1/(2*c), 1, 2*n)];
    Wc = Wm;
    Wc(1) = Wc(1) + (1 - alpha^2 + beta);
    Y = zeros(1, 2*n+1);% Propagate through measurement function
    for k = 1:(2*n+1)
        Y(k) = hfun(X(:,k));
    end

    yHat = sum(Wm .* Y);
    Syy = R;% Innovation variance Syy and cross-covariance Pxy
    Pxy = zeros(n,1);
    for k = 1:(2*n+1)
        dy = (Y(k) - yHat);
        dx = (X(:,k) - x);
        Syy = Syy + Wc(k) * (dy*dy);
        Pxy = Pxy + Wc(k) * (dx*dy);
    end

    % Kalman gain, update
    K = Pxy / Syy;
    xNew = x + K*(y - yHat);
    PNew = P - K*Syy*K';
end
function price = heston_price_from_state(xstate, Cflag, powi, K, S, r, Tau)

    v0    = exp(xstate(1));
    kappa = exp(xstate(2));
    theta = exp(xstate(3));
    sig   = exp(xstate(4));
    rho   = tanh(xstate(5));

    par = [v0, kappa, theta, sig, rho];   % MUST be 1x5 row

    price = opt_price('Heston', par, Cflag, powi, K, S, r, Tau);
end
function price = bates_price_from_jumpstate(xstate, V0, kappa, theta, sigma, rho, Cflag, powi, K, S, r, Tau)

%  Bates price from jump-state with fixed Heston part
    lambda  = exp(xstate(1));
    mu_j    = xstate(2);
    sigma_j = exp(xstate(3));

    parB = [V0, kappa, theta, sigma, rho, lambda, mu_j, sigma_j];  % 1x8 row
    price = opt_price('Bates', parB, Cflag, powi, K, S, r, Tau);
end

%% Section 2.8.a UKF on Bates: estimate ALL parameters par_t = [V0,kappa,theta,sigma,rho,lambda,mu_j,sigma_j]
% Bates par order for opt_price('Bates'):
% parB = [V0, kappa, theta, sigma, rho, lambda, mu_j, sigma_j]  (1x8 row)
%
% State in transformed space (enforce constraints):
% x = [log(V0); log(kappa); log(theta); log(sigma); atanh(rho); log(lambda); mu_j; log(sigma_j)]
% so:
%   V0,kappa,theta,sigma,lambda,sigma_j > 0 and rho in (-1,1), mu_j free

alphaUKF = 1e-3; 
betaUKF  = 2; 
kappaUKF = 0;
nB = 8;                         % full Bates state dimension
xHatB = zeros(nB, T);
PHatB = zeros(nB, nB, T);
%Initial guess, can tune
V0_0     = 0.04;     % initial variance 
kappa0   = 2.0;
theta0   = 0.04;
sigma0   = 0.5;
rho0     = -0.5;
lambda0  = 0.8;      % jump intensity
mu_j0    = -0.2;     % mean log jump
sigma_j0 = 0.20;     % jump std (log jump)

x  = [log(V0_0);
      log(kappa0);
      log(theta0);
      log(sigma0);
      atanh(rho0);
      log(lambda0);
      mu_j0;
      log(sigma_j0)];

% Prior covariance can tune
Pxx = diag([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.0, 0.8].^2);
% Process noise (random-walk drift per snapshot) 
% If the filter moves too wildly: decrease Q
% If it barely moves / fits poorly: increase Q
QB  = diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-3, 5e-2, 5e-2]);
% Measurement noise variance 
RB  = (0.5)^2;

for t = 1:T

    % Prediction (random walk)
    xPred = x;
    PPred = Pxx + QB;

    calIdx = idxCal{t};
    if isempty(calIdx)
        x = xPred; Pxx = PPred;
        xHatB(:,t)   = x;
        PHatB(:,:,t) = Pxx;
        continue
    end

    % Sequential updates noticed in your code style
    for j = 1:numel(calIdx)
        i = calIdx(j);

        yObs = opt(t).MID(i);

        Ci   = opt(t).C(i);
        Ki   = opt(t).K(i);
        Si   = opt(t).S(i);
        ri   = opt(t).r(i);
        Taui = opt(t).T(i);
        powi = 1;

        hfun = @(xstate) bates_price_from_fullstate(xstate, Ci, powi, Ki, Si, ri, Taui);

        [xPred, PPred] = ukf_update_scalar(xPred, PPred, yObs, hfun, RB, alphaUKF, betaUKF, kappaUKF);
    end

    % Save filtered
    x = xPred; Pxx = PPred;
    xHatB(:,t)   = x;
    PHatB(:,:,t) = Pxx;
end

%  Map state to parameter paths 
V0Hat_B     = exp(xHatB(1,:)).';
kappaHat_B  = exp(xHatB(2,:)).';
thetaHat_B  = exp(xHatB(3,:)).';
sigmaHat_B  = exp(xHatB(4,:)).';
rhoHat_B    = tanh(xHatB(5,:)).';
lambdaHat_B = exp(xHatB(6,:)).';
mu_jHat_B   = xHatB(7,:).';
sigma_jHat_B= exp(xHatB(8,:)).';

figure; 
plot(1:T, sqrt(V0Hat_B), 'LineWidth', 1.2); grid on;
xlabel('Time index (10-min snapshot)'); ylabel('\surd V_{0,t}');
title('Bates UKF: sqrt(V0_t)');
figure; 
plot(1:T, kappaHat_B, 'LineWidth', 1.2); hold on;
plot(1:T, thetaHat_B, 'LineWidth', 1.2);
plot(1:T, sigmaHat_B, 'LineWidth', 1.2);
plot(1:T, rhoHat_B, 'LineWidth', 1.2);
grid on;
xlabel('Time index (10-min snapshot)'); ylabel('Value');
title('Bates UKF: Heston-type parameters');
legend('\kappa','\theta','\sigma','\rho','Location','best');
figure;
plot(1:T, lambdaHat_B, 'LineWidth', 1.2); hold on;
plot(1:T, mu_jHat_B, 'LineWidth', 1.2);
plot(1:T, sigma_jHat_B, 'LineWidth', 1.2);
grid on;
xlabel('Time index (10-min snapshot)'); ylabel('Value');
title('Bates UKF: jump parameters');
legend('\lambda','\mu_j','\sigma_j','Location','best');

%% Section 2.9.a Hold-out pricing errors (Bates + UKF, ALL parameters estimated)

rmseB_in = NaN(T,1);
maeB_in  = NaN(T,1);
rmseB_1s = NaN(T-1,1);
maeB_1s  = NaN(T-1,1);

for t = 1:T

    holdIdx = idxHold{t};
    if isempty(holdIdx), continue; end

    % Bates parameters at time t from full UKF (Section 2.8.a)
    V0_t    = V0Hat_B(t);
    kappa_t = kappaHat_B(t);
    theta_t = thetaHat_B(t);
    sigma_t = sigmaHat_B(t);
    rho_t   = rhoHat_B(t);

    lam_t   = lambdaHat_B(t);
    muj_t   = mu_jHat_B(t);
    sigj_t  = sigma_jHat_B(t);

    parB_t = [V0_t, kappa_t, theta_t, sigma_t, rho_t, lam_t, muj_t, sigj_t];  % 1x8 row

    % Interpolation (same t)
    yObs = opt(t).MID(holdIdx);
    yHat = zeros(numel(holdIdx),1);
    for j = 1:numel(holdIdx)
        i = holdIdx(j);
        yHat(j) = opt_price('Bates', parB_t, opt(t).C(i), 1, opt(t).K(i), opt(t).S(i), opt(t).r(i), opt(t).T(i));
    end

    err = yObs - yHat;
    rmseB_in(t) = sqrt(mean(err.^2));
    maeB_in(t)  = mean(abs(err));

    % Time extrapolation t -> t+1 using parameters from time t
    if t < T
        holdIdx_next = idxHold{t+1};
        if ~isempty(holdIdx_next)

            yObs2 = opt(t+1).MID(holdIdx_next);
            yHat2 = zeros(numel(holdIdx_next),1);

            for j = 1:numel(holdIdx_next)
                i = holdIdx_next(j);
                yHat2(j) = opt_price('Bates', parB_t, opt(t+1).C(i), 1, opt(t+1).K(i), opt(t+1).S(i), opt(t+1).r(i), opt(t+1).T(i));
            end

            err2 = yObs2 - yHat2;
            rmseB_1s(t) = sqrt(mean(err2.^2));
            maeB_1s(t)  = mean(abs(err2));
        end
    end
end

figure;
plot(rmseB_in, 'LineWidth', 1.2); hold on;
plot([rmseB_1s; NaN], 'LineWidth', 1.2);
xlabel('Time index (10-min snapshot)');
ylabel('RMSE on hold-out options');
title('Bates (full UKF) RMSE: interpolation vs time extrapolation');
legend('Interpolation (same t)','Time extrapolation (t \rightarrow t+1)','Location','best');
grid on;
fprintf('\nBates(full UKF) Average RMSE (interpolation): %.6f\n', mean(rmseB_in,'omitnan'));
fprintf('Bates(full UKF) Average RMSE (time extrapolation): %.6f\n\n', mean(rmseB_1s,'omitnan'));
% Results are more or less the same as fixed Heston from 2.8
%% Section 2.10 RMSE by moneyness & maturity (quantile bins, Bates full UKF)

mEdges  = [0, 0.9, 0.97, 1.03, 1.10, inf];
mLabels = {'Deep ITM','ITM','ATM','OTM','Deep OTM'};
nMny = numel(mLabels);
nTau = 3;
SSE_B   = zeros(nMny, nTau);
COUNT_B = zeros(nMny, nTau);

for t = 1:T

    holdIdx = idxHold{t};
    if isempty(holdIdx), continue; end

    % Bates parameters at time t from full UKF (Section 2.8.a)
    V0_t    = V0Hat_B(t);
    kappa_t = kappaHat_B(t);
    theta_t = thetaHat_B(t);
    sigma_t = sigmaHat_B(t);
    rho_t   = rhoHat_B(t);

    lam_t   = lambdaHat_B(t);
    muj_t   = mu_jHat_B(t);
    sigj_t  = sigma_jHat_B(t);

    parB_t = [V0_t, kappa_t, theta_t, sigma_t, rho_t, lam_t, muj_t, sigj_t];  % 1x8 row

    % Hold-out data
    Pobs = opt(t).MID(holdIdx);
    K    = opt(t).K(holdIdx);
    S    = opt(t).S(holdIdx);
    Tau  = opt(t).T(holdIdx);
    C    = opt(t).C(holdIdx);
    r_t  = opt(t).r(holdIdx);

    m = K ./ S;
    tauEdges = quantile(Tau, [0 0.33 0.66 1]);
    tauEdges(end) = tauEdges(end) + 1e-12;

    % Model prices
    Phat = zeros(size(Pobs));
    for j = 1:numel(holdIdx)
        Phat(j) = opt_price('Bates', parB_t, C(j), 1, K(j), S(j), r_t(j), Tau(j));
    end

    err = Pobs - Phat;

    % Bin errors
    for a = 1:nMny
        inM = (m >= mEdges(a)) & (m < mEdges(a+1));
        for b = 1:nTau
            inT = (Tau >= tauEdges(b)) & (Tau < tauEdges(b+1));
            inBin = inM & inT;

            nb = sum(inBin);
            if nb > 0
                SSE_B(a,b)   = SSE_B(a,b) + sum(err(inBin).^2);
                COUNT_B(a,b) = COUNT_B(a,b) + nb;
            end
        end
    end
end

RMSE_B = NaN(nMny, nTau);
for a = 1:nMny
    for b = 1:nTau
        if COUNT_B(a,b) > 0
            RMSE_B(a,b) = sqrt(SSE_B(a,b) / COUNT_B(a,b));
        end
    end
end

RMSE_B_table  = array2table(RMSE_B,  'VariableNames', {'Short(Q1)','Medium(Q2)','Long(Q3)'}, 'RowNames', mLabels);
COUNT_B_table = array2table(COUNT_B,'VariableNames', {'Short(Q1)','Medium(Q2)','Long(Q3)'}, 'RowNames', mLabels);

disp('Bates(full UKF) RMSE by moneyness and maturity (quantile-based)');
disp(RMSE_B_table);
disp('Counts per bin');
disp(COUNT_B_table);
% Bates pricing from full Bates state
function price = bates_price_from_fullstate(xstate, Cflag, powi, K, S, r, Tau)
    V0      = exp(xstate(1));
    kappa   = exp(xstate(2));
    theta   = exp(xstate(3));
    sigma   = exp(xstate(4));
    rho     = tanh(xstate(5));
    lambda  = exp(xstate(6));
    mu_j    = xstate(7);
    sigma_j = exp(xstate(8));

    parB = [V0, kappa, theta, sigma, rho, lambda, mu_j, sigma_j];  % 1x8 row
    price = opt_price('Bates', parB, Cflag, powi, K, S, r, Tau);
end