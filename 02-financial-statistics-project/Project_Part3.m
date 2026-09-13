%% 3.1 Load data and plot returns 
clear; clc; close all
ASSETS = readtable('ASSETSA.csv');
Date   = ASSETS.Date;
PRICES = table2array(ASSETS(:,2:13));
LOGRETURNS = diff(log(PRICES));
FULLNAMES = { ...
'The Allstate Corporation (ALL)', ...
'Vanguard Total Bond Market Index Fund (BND)', ...
'Bitcoin USD Price (BTC-USD)', ...
'Intel Corporation (INTC)', ...
'Coffee futures (KC=F)', ...
'NextEra Energy, Inc. (NEE)', ...
'NVIDIA Corporation (NVDA)', ...
'Silver futures (SI=F)', ...
'Sysco Corporation (SYY)', ...
'UnitedHealth Group Incorporated (UNH)', ...
'Walmart Inc. (WMT)', ...
'Exxon Mobil Corporation (XOM)' };

figure(1)
for i = 1:12
    subplot(4,3,i)
    plot(Date, PRICES(:,i), 'LineWidth', 1.2)
    title(FULLNAMES{i}, 'Interpreter','none')
    grid on
end

figure(2)
for i = 1:12
    subplot(4,3,i)
    plot(Date(2:end), LOGRETURNS(:,i), 'LineWidth', 1.2)
    title(FULLNAMES{i}, 'Interpreter','none')
    grid on
end

%% 3.2 Set modelling data and evaluation data(saved and not used for modelling)

if ~isdatetime(Date)% Make sure Date is datetime
    try
        Date = datetime(Date);  
    catch
        Date = datetime(Date, 'InputFormat','yyyy-MM-dd'); 
    end
end

DateR = Date(2:end); 
inStart = datetime(2015,1,28);
inEnd   = datetime(2017,12,31);
idxIn  = (DateR >= inStart) & (DateR <= inEnd);
idxOut = (DateR >  inEnd);
R_in  = LOGRETURNS(idxIn, :);     % in-sample returns for modelling (2015-2017)
R_out = LOGRETURNS(idxOut, :);    % remaining returns for evaluation (2018-2020)
Date_in  = DateR(idxIn);
Date_out = DateR(idxOut);
fprintf('In-sample returns:  %d obs (%s to %s)\n', size(R_in,1),  string(Date_in(1)),  string(Date_in(end)));
fprintf('Out-of-sample ret.: %d obs (%s to %s)\n', size(R_out,1), string(Date_out(1)), string(Date_out(end)));


%% 3.3 Unconditional mean/cov from in-sample and data-driven gamma (NaN-safe)

d = size(R_in,2);  % number of assets =12
% Remove any days where at least one asset return is missing/non-finite
keep = all(isfinite(R_in), 2);
R_in_clean = R_in(keep, :);
Date_in_clean = Date_in(keep);
fprintf('\nR_in rows before/after cleaning: %d -> %d\n', size(R_in,1), size(R_in_clean,1));
fprintf('First/last clean in-sample date: %s to %s\n', string(Date_in_clean(1)), string(Date_in_clean(end)));
% Unconditional mean and covariance 
muHat    = mean(R_in_clean, 1)';     % d x 1
SigmaHat = cov(R_in_clean, 1);       % d x d
mu_bar     = mean(abs(muHat));
sigma2_bar = trace(SigmaHat) / d;
gammaHat = 2 * mu_bar / sigma2_bar;
fprintf('\nUnconditional estimates from 2015-2017 (cleaned):\n');
fprintf('mu_bar      = %.6g\n', mu_bar);
fprintf('sigma2_bar  = %.6g\n', sigma2_bar);
fprintf('gammaHat    = %.6g\n\n', gammaHat);
gamma = gammaHat;

%% 3.4 Fit heavy-tailed GARCH(1,1) for each of the 12 assets (2015-2017)

d = size(R_in_clean, 2);   
thetaHatS = NaN(6, d);     % [mu; omega; alpha; beta; cTail; fTail]
lnLS_all  = NaN(d, 1);
VCS_all   = NaN(6, 6, d);

for i = 1:d
    r_i = R_in_clean(:, i);

    %initial guess
    mu0    = mean(r_i);
    omega0 = 0.01 * var(r_i);
    alpha0 = 0.05;
    beta0  = 0.90;
    cTail0 = 0.20;
    fTail0 = 2;
    theta0 = [mu0; omega0; alpha0; beta0; cTail0; fTail0];

    
    try
        [thHat, lnL, VCV] = MLmax(@lnl_garch_spec, theta0, r_i);
         thetaHatS(:, i) = thHat(:);
        lnLS_all(i)     = lnL;
        VCS_all(:,:,i)  = VCV;

    catch ME
        fprintf('Asset %d failed in MLmax: %s\n', i, ME.message);

        % Optional fallback
        alpha0b = 0.10; beta0b = 0.85;
        theta0b = [mu0; omega0; alpha0b; beta0b; cTail0; fTail0];

        try
            [thHat, lnL, VCV] = MLmax(@lnl_garch_spec, theta0b, r_i);
            thetaHatS(:, i) = thHat(:);
            lnLS_all(i)     = lnL;
            VCS_all(:,:,i)  = VCV;
        catch
            fprintf('  Fallback also failed for asset %d.\n', i);
        end
    end

    fprintf('Asset %2d done. lnL = %.4f\n', i, lnLS_all(i));
end

muS    = thetaHatS(1,:).';
omegaS = thetaHatS(2,:).';
alphaS = thetaHatS(3,:).';
betaS  = thetaHatS(4,:).';
cTailS = thetaHatS(5,:).';
fTailS = thetaHatS(6,:).';

Results_GARCHspec = table(muS, omegaS, alphaS, betaS, cTailS, fTailS, lnLS_all,  'VariableNames', {'mu','omega','alpha','beta','cTail','fTail','lnL'}, ...
    'RowNames', FULLNAMES);
disp('Heavy-tailed GARCH(1,1) estimates (2015-2017, per asset):');
disp(Results_GARCHspec);


%% 3.5 Try Fixed Omega 

d = size(R_in_clean, 2);
omegaFixVec = [  5.0986e-07;  2.2632e-07;1e-5; 7.9746e-06;  1.1565e-06; 1.1886e-06;  1e-06;  1e-06;   2.0413e-05;  1.9510e-06; 1e-05;  1.6898e-06 ];
thetaHatS = NaN(5, d);        
lnLS_all  = NaN(d, 1);
VCS_all   = NaN(5, 5, d);
epsS = 1e-6;

for i = 1:d
    r_i = R_in_clean(:, i);
    omegaFix = omegaFixVec(i);
    mu0 = mean(r_i);

    if i == 3
        % BTC: unconstrained parameters u = [mu; a; b; u; v]
        a0 = log(0.05);  b0 = log(0.90);  u0 = 0;  v0 = log(2);
        theta0  = [mu0; a0; b0; u0; v0];

        try
            [thHat, lnL, VCV] = MLmax(@(par,data) lnl_garch_spec_fixomega_BTC_stationary(par, data, omegaFix, epsS), theta0, r_i);
        catch
            a0b = log(0.10); b0b = log(0.85);
            theta0b = [mu0; a0b; b0b; u0; v0];
            [thHat, lnL, VCV] = MLmax(@(par,data) lnl_garch_spec_fixomega_BTC_stationary(par, data, omegaFix, epsS), theta0b, r_i);
        end

        % Print CI for transformed BTC state (
        paramNamesU = {'mu','a','b','u','v'};
        CI_U = waldCI_table(thHat, VCV, paramNamesU);
        fprintf('\n[3.5 BTC] Asset %2d (fixed omega) transformed estimates + 95%% CI:\n', i);
        disp(CI_U);
        fprintf('omegaFix = %.3g, lnL = %.4f\n', omegaFix, lnL);

        % Map to economically interpretable point estimates
        [alphaHat, betaHat, cTailHat, fTailHat] = btc_map(thHat, epsS);
        fprintf('Mapped (point): alpha=%.4f, beta=%.4f, cTail=%.4f, fTail=%.4f\n\n',  alphaHat, betaHat, cTailHat, fTailHat);

        thetaHatS(:, i) = [thHat(1); alphaHat; betaHat; cTailHat; fTailHat];
        lnLS_all(i)     = lnL;
        VCS_all(:,:,i)  = VCV;

    else
        
        alpha0 = 0.05; beta0 = 0.90; cTail0 = 0.20; fTail0 = 2.0;
        theta0 = [mu0; alpha0; beta0; cTail0; fTail0];

        try
            [thHat, lnL, VCV] = MLmax(@(par,data) lnl_garch_spec_fixomega(par, data, omegaFix), theta0, r_i);
        catch
            alpha0b = 0.10; beta0b = 0.85;
            theta0b = [mu0; alpha0b; beta0b; cTail0; fTail0];
            [thHat, lnL, VCV] = MLmax(@(par,data) lnl_garch_spec_fixomega(par, data, omegaFix), theta0b, r_i);
        end

        % Print CI (dimension matches = 5)
        paramNames = {'mu','alpha','beta','cTail','fTail'};
        CI_T = waldCI_table(thHat, VCV, paramNames);
        fprintf('\n[3.5] Asset %2d (fixed omega) estimates + 95%% CI:\n', i);
        disp(CI_T);
        fprintf('omegaFix = %.3g, lnL = %.4f\n\n', omegaFix, lnL);

        thetaHatS(:, i) = thHat(:);
        lnLS_all(i)     = lnL;
        VCS_all(:,:,i)  = VCV;
    end

    fprintf('Asset %2d done. omegaFix=%.3g, lnL=%.4f\n', i, omegaFix, lnLS_all(i));
end

muS    = thetaHatS(1,:).';
alphaS = thetaHatS(2,:).';
betaS  = thetaHatS(3,:).';
cTailS = thetaHatS(4,:).';
fTailS = thetaHatS(5,:).';
Results = table(muS, omegaFixVec, alphaS, betaS, cTailS, fTailS, lnLS_all,  'VariableNames', {'mu','omegaFix','alpha','beta','cTail','fTail','lnL'},'RowNames', FULLNAMES);
disp(Results);

%% 3.6 Diagnostics in big tiled figures (QQ, ACF(z), ACF(z^2))

d = size(R_in_clean,2);
T = size(R_in_clean,1);
Z = NaN(T,d);

for i = 1:d
    r_i = R_in_clean(:,i);

    muHat_i    = thetaHatS(1,i);
    alphaHat_i = thetaHatS(2,i);
    betaHat_i  = thetaHatS(3,i);

    omegaFix = omegaFixVec(i);
    eps_i = r_i - muHat_i;

    h_i = zeros(T,1);
    h_i(1) = max(var(r_i), 1e-12);

    for t = 2:T
        h_i(t) = omegaFix + alphaHat_i*eps_i(t-1)^2 + betaHat_i*h_i(t-1);
        if ~(isfinite(h_i(t))) || h_i(t) <= 0
            h_i(t) = NaN;
        end
    end

    Z(:,i) = eps_i ./ sqrt(h_i);
end

noLags = 30;

% Figure 1: QQ plots 
figure;
tl = tiledlayout(4,3,'Padding','compact','TileSpacing','compact');
title(tl,'QQ plots of standardized residuals z_t (fixed-omega heavy-tailed GARCH)','Interpreter','none');

for i = 1:d
    nexttile;
    z = Z(:,i);
    z = z(isfinite(z));
    if numel(z) < 20
        text(0.1,0.5,'Too few finite z','Units','normalized'); axis off;
    else
        qqplot(z);
        grid on;
        title(strrep(FULLNAMES{i},'_','\_'),'Interpreter','none','FontSize',8);
    end
end


acf_fun = @(x,L) local_acf(x,L);
% Figure 2: ACF of z_t with 95% confidence bands
figure;
tl = tiledlayout(4,3,'Padding','compact','TileSpacing','compact');
title(tl,sprintf('ACF of standardized residuals z_t (lags 0..%d)',noLags),'Interpreter','none');

for i = 1:d
    nexttile;
    z = Z(:,i);
    z = z(isfinite(z));
    T_i = numel(z);

    if T_i < (noLags+5)
        text(0.1,0.5,'Too few finite z','Units','normalized');
        axis off;
    else
        acf = acf_fun(z,noLags);
        ci  = 1.96 / sqrt(T_i);

        stem(0:noLags, acf, 'filled'); hold on;
        yline(0,'k-');
        yline(ci,'r--');
        yline(-ci,'r--');
        hold off;

        grid on;
        title(strrep(FULLNAMES{i},'_','\_'),'Interpreter','none','FontSize',8);
        xlim([0 noLags]);
        ylim([-0.25 0.25]);
    end
end

% Figure 3: ACF of z_t^2 with 95% confidence bands
figure;
tl = tiledlayout(4,3,'Padding','compact','TileSpacing','compact');
title(tl,sprintf('ACF of squared standardized residuals z_t^2 (lags 0..%d)',noLags),'Interpreter','none');

for i = 1:d
    nexttile;
    z = Z(:,i);
    z = z(isfinite(z));
    T_i = numel(z);

    if T_i < (noLags+5)
        text(0.1,0.5,'Too few finite z','Units','normalized');
        axis off;
    else
        z2  = z.^2;
        acf2 = acf_fun(z2,noLags);
        ci   = 1.96 / sqrt(T_i);

        stem(0:noLags, acf2, 'filled'); hold on;
        yline(0,'k-');
        yline(ci,'r--');
        yline(-ci,'r--');
        hold off;

        grid on;
        title(strrep(FULLNAMES{i},'_','\_'),'Interpreter','none','FontSize',8);
        xlim([0 noLags]);
        ylim([-0.25 0.25]);
    end
end

%% 3.7 Univariate EGARCH(1,1) with heavy-tailed lnl_spec innovations  (2015–2017)
% Model (standard EGARCH(1,1)):
%   eps_t = r_t - mu
%   z_t   = eps_t / sqrt(h_t)
%   log h_t = omega + beta*log h_{t-1} + alpha*(|z_{t-1}| - E|Z|) + gamma*z_{t-1}
% We use E|Z| = sqrt(2/pi) (Gaussian benchmark) as the usual centering constant.
%
% Innovations density: eps_t | F_{t-1} ~ lnl_spec(cTail,fTail, scale = sqrt(h_t))
%
% Parameters per asset:
%   theta = [mu, omega, alpha, gamma, beta, cTail, fTail]'

d = size(R_in_clean, 2);
T = size(R_in_clean, 1);

thetaHatE = NaN(7, d);      % [mu; omega; alpha; gamma; beta; cTail; fTail]
lnLE_all  = NaN(d, 1);
VCE_all   = NaN(7, 7, d);
% Filter outputs 
EPS_E  = NaN(T, d);
SIG2_E = NaN(T, d);
Z_E    = NaN(T, d);

for i = 1:d
    r_i = R_in_clean(:, i);

    %Initial guesses 
    mu0   = mean(r_i);
    v0    = var(r_i);
    beta0 = 0.95;               % persistence (|beta|<1)
    omega0 = (1-beta0)*log(max(v0,1e-12));  
    alpha0 = 0.10;              % magnitude effect
    gamma0 = 0.00;              % leverage/asymmetry
    cTail0 = 0.20;              
    fTail0 = 2.00;              

    theta0 = [mu0; omega0; alpha0; gamma0; beta0; cTail0; fTail0];

    % ML estimation via MLmax 
    try
        [thHat, lnL, VCV] = MLmax(@lnl_egarch_spec, theta0, r_i);
        thetaHatE(:, i) = thHat(:);
        lnLE_all(i)     = lnL;
        VCE_all(:,:,i)  = VCV;
    catch ME
        fprintf('EGARCH asset %d failed in MLmax: %s\n', i, ME.message);

        % fallback guess
        beta0b  = 0.90;
        omega0b = (1-beta0b)*log(max(v0,1e-12));
        theta0b = [mu0; omega0b; 0.15; 0.00; beta0b; cTail0; fTail0];

        try
            [thHat, lnL, VCV] = MLmax(@lnl_egarch_spec, theta0b, r_i);
            thetaHatE(:, i) = thHat(:);
            lnLE_all(i)     = lnL;
            VCE_all(:,:,i)  = VCV;
        catch ME2
            fprintf('  Fallback also failed for EGARCH asset %d: %s\n', i, ME2.message);
        end
    end

    fprintf('EGARCH asset %2d done. lnL = %.4f\n', i, lnLE_all(i));

    % Filter with estimated parameters to get eps_t, h_t, z_t 
    if all(isfinite(thetaHatE(:,i)))
        [eps_i, h_i, z_i] = egarch_filter_spec(r_i, thetaHatE(:,i));
        EPS_E(:,i)  = eps_i;
        SIG2_E(:,i) = h_i;
        Z_E(:,i)    = z_i;
    end
end

% Results table
muE    = thetaHatE(1,:).';
omegaE = thetaHatE(2,:).';
alphaE = thetaHatE(3,:).';
gammaE = thetaHatE(4,:).';
betaE  = thetaHatE(5,:).';
cTailE = thetaHatE(6,:).';
fTailE = thetaHatE(7,:).';
Results_EGARCHspec = table(muE, omegaE, alphaE, gammaE, betaE, cTailE, fTailE, lnLE_all,  'VariableNames', {'mu','omega','alpha','gamma','beta','cTail','fTail','lnL'}, ...
    'RowNames', FULLNAMES);
disp('Heavy-tailed EGARCH(1,1) + lnl_spec estimates (2015-2017, per asset):');
disp(Results_EGARCHspec);

%EGARCH loglik using lnl_spec 
function l = lnl_egarch_spec(theta, r)
% theta = [mu, omega, alpha, gamma, beta, cTail, fTail]'
% EGARCH recursion:
%   log h_t = omega + beta log h_{t-1} + alpha(|z_{t-1}| - E|Z|) + gamma z_{t-1}
% z_{t-1} = eps_{t-1}/sqrt(h_{t-1})
% innovations density via lnl_spec with scale sqrt(h_t)

    mu    = theta(1);
    omega = theta(2);
    alpha = theta(3);
    gamma = theta(4);
    beta  = theta(5);
    cTail = theta(6);
    fTail = theta(7);

    r = r(:);
    T = length(r);
    l = NaN(T,1);

    % constraints 
    % Stationarity/persistence in EGARCH: |beta| < 1
    if ~isfinite(beta) || abs(beta) >= 0.999999
        l = -1e10*ones(T,1); return
    end
    % Tail params for lnl_spec
    if ~isfinite(cTail) || ~isfinite(fTail) || (cTail < 0) || (cTail >= 1) || (fTail <= 0)
        l = -1e10*ones(T,1); return
    end
 

    eps = r - mu;

    %EGARCH recursion in log-variance
    logh = zeros(T,1);
    h    = zeros(T,1);

    % init at sample var
    h(1)    = max(var(r), 1e-12);
    logh(1) = log(h(1));

    EabsZ = sqrt(2/pi);  % standard centering constant

    for t = 2:T
        z_prev = eps(t-1) / sqrt(h(t-1));
        logh(t) = omega + beta*logh(t-1) + alpha*(abs(z_prev) - EabsZ) + gamma*z_prev;

        
        if ~isfinite(logh(t))
            l = -1e10*ones(T,1); return
        end

        h(t) = exp(logh(t));
        if ~isfinite(h(t)) || h(t) <= 0
            l = -1e10*ones(T,1); return
        end
    end

    s_t   = sqrt(h);
    parHF = [cTail; fTail];

    
    l = lnl_spec(parHF, s_t, eps);

    % If lnl_spec returned NaNs, penalize
    if any(~isfinite(l))
        l = -1e10*ones(T,1);
    end
end
function [eps, h, z] = egarch_filter_spec(r, thetaHat)
    mu    = thetaHat(1);
    omega = thetaHat(2);
    alpha = thetaHat(3);
    gamma = thetaHat(4);
    beta  = thetaHat(5);

    r = r(:);
    T = length(r);

    eps = r - mu;
    h   = zeros(T,1);
    z   = zeros(T,1);

    h(1) = max(var(r), 1e-12);
    z(1) = eps(1) / sqrt(h(1));

    logh = zeros(T,1);
    logh(1) = log(h(1));

    EabsZ = sqrt(2/pi);

    for t = 2:T
        z_prev = eps(t-1) / sqrt(h(t-1));
        logh(t) = omega + beta*logh(t-1) + alpha*(abs(z_prev) - EabsZ) + gamma*z_prev;
        h(t) = exp(logh(t));
        h(t) = max(h(t), 1e-12);
        z(t) = eps(t) / sqrt(h(t));
    end
end

%%  3.8 CCC-GARCH for all 12 assets using fixed-omega GARCH(1,1) estimates (QML)
% We use the fixed-omega heavy-tailed GARCH(1,1) estimates from Section 3.5:
% thetaHatS(:,i) = [mu_i; alpha_i; beta_i; cTail_i; fTail_i]
% omegaFixVec(i) = fixed omega per asset
d = size(R_in_clean,2);
T = size(R_in_clean,1);
muVec    = thetaHatS(1,:).';   
alphaVec = thetaHatS(2,:).';  
betaVec  = thetaHatS(3,:).';   
omegaVec = omegaFixVec(:);     
% univariate filtering to get eps_t and sigma2_t for each asset
EPS    = NaN(T,d);
SIG2   = NaN(T,d);
Z      = NaN(T,d);

for i = 1:d
    r_i = R_in_clean(:,i);

    mu_i    = muVec(i);
    om_i    = omegaVec(i);
    a_i     = alphaVec(i);
    b_i     = betaVec(i);

    % residuals
    eps_i = r_i - mu_i;

    % variance recursion (fixed omega)
    h_i = zeros(T,1);
    h_i(1) = max(var(r_i), 1e-12);

    for t = 2:T
        h_i(t) = om_i + a_i*eps_i(t-1)^2 + b_i*h_i(t-1);
        if ~(isfinite(h_i(t))) || h_i(t) <= 0
            h_i(t) = NaN;  % keep NaN and let pairwise correlation handle it
        end
    end

    EPS(:,i)  = eps_i;
    SIG2(:,i) = h_i;
    Z(:,i)    = eps_i ./ sqrt(h_i);
end

% estimate constant correlation from standardized residuals
Rhat = corr(Z, 'Rows','pairwise');
Rhat = 0.5*(Rhat + Rhat.');
[Rhat_pd, jitterUsed] = makePD(Rhat);
fprintf('CCC step: Rhat PD enforced with jitter = %.3g\n', jitterUsed);
% compute Gaussian CCC log-likelihood (QML)
lnL_ccc = ccc_loglik_gauss(EPS, SIG2, Rhat_pd);
fprintf('CCC-GARCH (Gaussian QML) total lnL = %.4f\n', sum(lnL_ccc(isfinite(lnL_ccc))));
Rhat_table = array2table(Rhat_pd,  'RowNames', FULLNAMES, 'VariableNames', FULLNAMES);
disp('Estimated constant correlation matrix Rhat (CCC):');
disp(Rhat_table);
% Store
CCC = struct();
CCC.muVec = muVec;
CCC.omegaFixVec = omegaVec;
CCC.alphaVec = alphaVec;
CCC.betaVec  = betaVec;
CCC.Rhat = Rhat_pd;
CCC.lnL_t = lnL_ccc;
CCC.lnL_total = sum(lnL_ccc(isfinite(lnL_ccc)));

%% 3.9 Sanity checks for CCC-GARCH (all assets, all pairs)
fprintf('\n=== CCC-GARCH diagnostics ===\n');
% How close is empirical corr(Z) to Rhat? 
R_sample = corr(Z, 'Rows', 'pairwise');   
DiffR = R_sample - Rhat_pd;
% Ignore diagonal for summaries
maskOff = ~eye(d);
maxAbsDiff = max(abs(DiffR(maskOff)));
meanAbsDiff = mean(abs(DiffR(maskOff)));
rmseDiff = sqrt(mean(DiffR(maskOff).^2));
fprintf('Corr(Z) vs Rhat (off-diagonal):\n');
fprintf('  max |diff|  = %.4f\n', maxAbsDiff);
fprintf('  mean|diff|  = %.4f\n', meanAbsDiff);
fprintf('  RMSE(diff)  = %.4f\n', rmseDiff);

% Whiten residuals using CCC: U = C^{-1} Z -
% If CCC is correct: cov(U) ≈ I, and U should be roughly serially uncorrelated
C = chol(Rhat_pd, 'lower');     % R = C*C'
U = (C \ Z')';                  % T x d
SigmaU = cov(U, 1);
Frob = norm(SigmaU - eye(d), 'fro');
offdiag = SigmaU - diag(diag(SigmaU));
maxOffU = max(abs(offdiag(:)));
fprintf('\nWhitened residuals U = C^{-1}Z:\n');
fprintf('  ||cov(U)-I||_F = %.4f\n', Frob);
fprintf('  max |offdiag(cov(U))| = %.4f\n', maxOffU);

% Ljung–Box on all components of U 
% Use same L for all; if many components reject in u or u^2 -> remaining dynamics 
L = 20;
p_u  = NaN(d,1);
p_u2 = NaN(d,1);

for k = 1:d
    u = U(:,k);
    u = u(isfinite(u));
    if numel(u) > (L+5)
        [~,p_u(k)]  = lbqtest(u,   'Lags', L);
        [~,p_u2(k)] = lbqtest(u.^2,'Lags', L);
    end
end

fracReject_u  = mean(p_u  < 0.05, 'omitnan');
fracReject_u2 = mean(p_u2 < 0.05, 'omitnan');

fprintf('\nLBQ on U components (L=%d):\n', L);
fprintf('  fraction rejecting LBQ(u)  at 5%% = %.2f\n', fracReject_u);
fprintf('  fraction rejecting LBQ(u^2) at 5%% = %.2f\n', fracReject_u2);
% print which components reject 
badU  = find(p_u  < 0.05);
badU2 = find(p_u2 < 0.05);
if ~isempty(badU)
    fprintf('  Components rejecting LBQ(u):  %s\n', mat2str(badU'));
end
if ~isempty(badU2)
    fprintf('  Components rejecting LBQ(u^2): %s\n', mat2str(badU2'));
end

%Time-variation of correlations: rolling corr for ALL pairs 

w = 250;                       % rolling window length
minObs = 0.9*w;               
pairIdx = find(triu(ones(d),1));
nPairs = numel(pairIdx);
rollStd = NaN(nPairs,1);       % std of rolling correlation per pair
rollMeanAbsDev = NaN(nPairs,1);% mean absolute deviation from Rhat per pair
pairI = NaN(nPairs,1);
pairJ = NaN(nPairs,1);

fprintf('\nRolling-correlation diagnostics (window=%d): computing %d pairs...\n', w, nPairs);

cnt = 0;
for ii = 1:d-1
    for jj = ii+1:d
        cnt = cnt + 1;
        pairI(cnt) = ii; pairJ(cnt) = jj;

        z_i = Z(:,ii);
        z_j = Z(:,jj);

        rho = NaN(T-w+1,1);
        for t = 1:T-w+1
            a = z_i(t:t+w-1);
            b = z_j(t:t+w-1);
            ok = isfinite(a) & isfinite(b);
            if sum(ok) >= minObs
                rho(t) = corr(a(ok), b(ok));
            end
        end

        % Rolling variability + deviation from CCC target
        rollStd(cnt)        = std(rho, 'omitnan');
        rollMeanAbsDev(cnt) = mean(abs(rho - Rhat_pd(ii,jj)), 'omitnan');
    end
end

% Summaries
fprintf('Rolling-corr summary across pairs:\n');
fprintf('  median std(roll corr)        = %.4f\n', median(rollStd,'omitnan'));
fprintf('  90th pct std(roll corr)      = %.4f\n', prctile(rollStd,90));
fprintf('  median mean|rho_t - Rhat|    = %.4f\n', median(rollMeanAbsDev,'omitnan'));
fprintf('  90th pct mean|rho_t - Rhat|  = %.4f\n', prctile(rollMeanAbsDev,90));
% Flag most time-varying pairs 
[~,ord] = sort(rollStd, 'descend');
topK = min(10, nPairs);
fprintf('\nTop %d pairs by rolling std (potential DCC evidence):\n', topK);
for k = 1:topK
    ii = pairI(ord(k)); jj = pairJ(ord(k));
    nameI = sprintf('%d',ii); nameJ = sprintf('%d',jj);
    if exist('FULLNAMES','var') && numel(FULLNAMES)==d
        nameI = FULLNAMES{ii}; nameJ = FULLNAMES{jj};
    end
    fprintf('  (%d,%d) std=%.4f  mean|dev|=%.4f  Rhat=%.4f   %s  vs  %s\n', ...
        ii, jj, rollStd(ord(k)), rollMeanAbsDev(ord(k)), Rhat_pd(ii,jj), nameI, nameJ);
end

% Plot a few most time-varying pairs 
figure; tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
titleStr = sprintf('Rolling correlations for top varying pairs (w=%d)', w);
sgtitle(titleStr,'Interpreter','none');

for p = 1:min(4,topK)
    ii = pairI(ord(p)); jj = pairJ(ord(p));

    z_i = Z(:,ii); z_j = Z(:,jj);
    rho = NaN(T-w+1,1);
    for t = 1:T-w+1
        a = z_i(t:t+w-1); b = z_j(t:t+w-1);
        ok = isfinite(a) & isfinite(b);
        if sum(ok) >= minObs
            rho(t) = corr(a(ok), b(ok));
        end
    end

    nexttile;
    plot(rho,'LineWidth',1); grid on;
    yline(Rhat_pd(ii,jj),'--');
    ttl = sprintf('Pair (%d,%d): Rhat=%.3f, std=%.3f', ii, jj, Rhat_pd(ii,jj), rollStd(ord(p)));
    title(ttl,'Interpreter','none');
end

fprintf('\n=== End CCC diagnostics ===\n');


%While CCC-GARCH adequately captures marginal volatility dynamics, rolling-window diagnostics reveal moderate to strong time variation in several pairwise correlations, suggesting that a DCC specification may be more appropriate if dynamic dependence is of interest.


%% Section 3.10  DCC(1,1) estimation on standardized residuals Z  (Gaussian QML)

keepZ = all(isfinite(Z),2);
Zc = Z(keepZ,:);
[Tc,d] = size(Zc);
% DCC initial guesses 
a0 = 0.05;
b0 = 0.90;
theta0_dcc = [log(a0); log(b0/(1-b0))];   % unconstrained parametrization
[th_dcc_u, lnL_dcc, V_dcc] = MLmax(@(u,data) lnl_dcc_gauss_constr(u,data), theta0_dcc, Zc);
% Map to constrained a,b
[aHat,bHat] = dcc_map(th_dcc_u);
paramNamesU = {'u1=log(a)','u2=logit(b/(1-b))'};
CI_U = waldCI_table(th_dcc_u, V_dcc, paramNamesU);
disp('[3.10] DCC unconstrained parameters + 95% CI:');
disp(CI_U);
fprintf('Implied aHat=%.4f, bHat=%.4f, a+b=%.4f\n\n', aHat, bHat, aHat+bHat);
fprintf('DCC(1,1) estimates: a=%.4f, b=%.4f, a+b=%.4f\n', aHat, bHat, aHat+bHat);
fprintf('DCC Gaussian QML lnL (corr part) = %.4f\n', lnL_dcc);
[Rseq,Qseq] = dcc_filter(Zc, aHat, bHat);

% Plot covariances between each pairs
pairs = [];  
for i = 1:d-1
    for j = i+1:d
        pairs = [pairs; i j];
    end
end
nPairs = size(pairs,1);
plotsPerFig = 12;
nFigs = ceil(nPairs / plotsPerFig);
tAxis = (1:Tc)';  
fprintf('Plotting %d DCC pairs in %d figures (12 subplots each)...\n', nPairs, nFigs);

for f = 1:nFigs
    idxStart = (f-1)*plotsPerFig + 1;
    idxEnd   = min(f*plotsPerFig, nPairs);
    thisIdx  = idxStart:idxEnd;

    figure;
    tiledlayout(4,3,'Padding','compact','TileSpacing','compact');
    sgtitle(sprintf('DCC Dynamic Correlations (pairs %d–%d of %d)', idxStart, idxEnd, nPairs), ...
        'Interpreter','none');

    for k = 1:numel(thisIdx)
        p = thisIdx(k);
        i = pairs(p,1);
        j = pairs(p,2);

        rho_t = squeeze(Rseq(i,j,:));  % Tc x 1

        nexttile;
        plot(tAxis, rho_t, 'LineWidth', 1); grid on;
        yline(0, ':');

        %  overlay CCC constant correlation
        if exist('Rhat_pd','var') && ~isempty(Rhat_pd)
            yline(Rhat_pd(i,j), '--');
        end

        title(sprintf('(%d,%d) %s vs %s', i, j, FULLNAMES{i}, FULLNAMES{j}), ...
            'Interpreter','none', 'FontSize', 7);
        xlabel('t'); ylabel('\rho_{t}');
        ylim([-1 1]);
    end


    for k = numel(thisIdx)+1:plotsPerFig
        nexttile; axis off;
    end
end
% LR test: CCC vs DCC correlation dynamics (2 extra params)by compare the correlation-part lnL only.
% use corr-part lnL for CCC: sum(-0.5*(logdet(Rhat)+z'Rhat^{-1}z)).
lnL_ccc_corr = sum( lnl_ccc_corr_only(Zc, Rhat_pd) );  
LR = 2*(lnL_dcc - lnL_ccc_corr);
pLR = 1 - chi2cdf(LR, 2);
fprintf('LR test CCC vs DCC (df=2): LR=%.4f, p=%.4g\n', LR, pLR);



%% 3.11 Out-of-sample portfolio evaluation (weekly rebalancing, 3-day horizon)
% We use the unused data R_out (2018-end) for evaluation.
% Strategy A: Dynamic MV portfolio using CCC-GARCH (rebalanced 3 days).
% Strategy B: Buy-and-hold benchmark from static (muHat, SigmaHat) estimated in-sample.
% Initial wealth W0 = 1,000,000.

W0 = 1e6;
gamma=gammaHat;
R_eval = R_out;              
Date_eval = Date_out;
[T_out, d] = size(R_eval);
% rebalancing every n trading days
rebStep = 3;
%Markowitz Constraints 
useLongOnly = true;          % no negative weights
useBudget1  = true;          % sum w = 1
% Regularization to avoid extreme weights when Sigma is ill-conditioned
ridge = 1e-8;
% Shorting not allowed
proj_simplex = @(w) project_to_simplex(w);

%  (A) Dynamic CCC-GARCH strategy 
W_dyn = NaN(T_out+1,1);
W_dyn(1) = W0;
nReb = floor((T_out-1)/rebStep) + 1;
Wgt_dyn = NaN(nReb, d);
RebDates = NaT(nReb,1);

% Initialize CCC recursion states using last in-sample day
% Use last in-sample EPS and SIG2 from section 3.8 as starting states:

eps_prev = EPS(end,:).';      % d x 1
h_prev   = SIG2(end,:).';     % d x 1

muVec = CCC.muVec(:);
omegaFixVec = CCC.omegaFixVec(:);
alphaVec = CCC.alphaVec(:);
betaVec  = CCC.betaVec(:);
Rhat = CCC.Rhat;

% Loop over out-sample days
rebIdx = 0;
w_curr = ones(d,1)/d;  
for t = 1:T_out

    % Rebalance at t = 1, 1+rebStep, ...
    if t == 1 || mod(t-1, rebStep) == 0
        rebIdx = rebIdx + 1;
        RebDates(rebIdx) = Date_eval(t);
        h_fc = omegaFixVec + alphaVec .* (eps_prev.^2) + betaVec .* h_prev;
        h_fc = max(h_fc, 1e-12);

        % Conditional covariance forecast
        D = diag(sqrt(h_fc));
        Sigma_fc = D * Rhat * D;

        % Conditional mean forecast:
        
        mu_fc = muVec;

        % Mean-variance weights 
        % Maximize: w' mu - (gamma/2) w' Sigma w
        
        Sigma_fc = Sigma_fc + ridge*eye(d);

        w_star = (1/gamma) * (Sigma_fc \ mu_fc);

        % Budget constraint sum w = 1: normalize
        if useBudget1
            if abs(sum(w_star)) < 1e-12
                w_star = ones(d,1)/d;
            else
                w_star = w_star / sum(w_star);
            end
        end

        % Long-only 
        if useLongOnly
            w_star = proj_simplex(w_star);
        end

        w_curr = w_star;
        Wgt_dyn(rebIdx,:) = w_curr.';
    end

    % return
    r_t = R_eval(t,:).';                 
    rp_t = w_curr' * r_t;                

    % Wealth update
    W_dyn(t+1) = W_dyn(t) * exp(rp_t);

    %  Update CCC states using realized eps_t and h_t 
    eps_t = r_t - muVec;
    h_t = omegaFixVec + alphaVec .* (eps_prev.^2) + betaVec .* h_prev;
    h_t = max(h_t, 1e-12);

    eps_prev = eps_t;
    h_prev   = h_t;
end

Wgt_dyn = Wgt_dyn(1:rebIdx,:);
RebDates = RebDates(1:rebIdx);

% Buy-and-hold benchmark with static Markowitz weights
% Use in-sample muHat and SigmaHat to compute one set of weights, then hold.
d = size(R_in_clean,2);
mu_static    = mean(R_in_clean, 1)';   % 12x1
Sigma_static = cov(R_in_clean, 1);     % 12x12
Sigma_static = Sigma_static + ridge*eye(d);
Sigma_static = 0.5*(Sigma_static + Sigma_static.');
w_bh = (1/gamma) * (Sigma_static \ mu_static);
% ridge for SPD / stability
Sigma_static = Sigma_static + ridge*eye(d);
% if SigmaHat has numerical issues, enforce symmetry + tiny jitter
Sigma_static = 0.5*(Sigma_static + Sigma_static.');
[~,p] = chol(Sigma_static);
if p ~= 0
    Sigma_static = Sigma_static + 1e-6*eye(d);
end

w_bh = (1/gamma) * (Sigma_static \ mu_static);

if useBudget1
    w_bh = w_bh / sum(w_bh);
end
if useLongOnly
    w_bh = proj_simplex(w_bh);
end

W_bh = NaN(T_out+1,1);
W_bh(1) = W0;
for t = 1:T_out
    rp_t = w_bh' * R_eval(t,:).';
    W_bh(t+1) = W_bh(t) * exp(rp_t);
end

%  Plot wealth paths 
figure;
plot([Date_eval(1); Date_eval], W_dyn, 'LineWidth', 1.5); hold on;
plot([Date_eval(1); Date_eval], W_bh,  'LineWidth', 1.5);
grid on; legend('CCC-GARCH weekly rebalance','Buy-and-hold (static)');
title('Out-of-sample wealth evolution (W0 = 1,000,000)','Interpreter','none');
xlabel('Date'); ylabel('Wealth');
%Performance metrics 
ret_dyn = diff(log(W_dyn));    % daily log returns of strategy
ret_bh  = diff(log(W_bh));
annFactor = 252;
mu_dyn = mean(ret_dyn,'omitnan');
sd_dyn = std(ret_dyn,'omitnan');
sh_dyn = (mu_dyn/sd_dyn)*sqrt(annFactor);
mu_bh = mean(ret_bh,'omitnan');
sd_bh = std(ret_bh,'omitnan');
sh_bh = (mu_bh/sd_bh)*sqrt(annFactor);
fprintf('\n=== Out-of-sample performance (daily log-return stats) ===\n');
fprintf('CCC weekly: mean=%.4g, sd=%.4g, Sharpe=%.3f, final wealth=%.2f\n', ...
    mu_dyn, sd_dyn, sh_dyn, W_dyn(end));
fprintf('Buy&Hold:  mean=%.4g, sd=%.4g, Sharpe=%.3f, final wealth=%.2f\n', ...
    mu_bh, sd_bh, sh_bh, W_bh(end));

% Show weights
W_bh_table = array2table(w_bh.', 'VariableNames', FULLNAMES);
disp('Buy-and-hold weights (static MV):');
disp(W_bh_table);

W_dyn_table = array2table(Wgt_dyn, 'VariableNames', FULLNAMES);
W_dyn_table.Date = RebDates;
W_dyn_table = movevars(W_dyn_table,'Date','Before',1);
disp('Dynamic CCC weekly weights (at each rebalance date):');
disp(W_dyn_table);


%% 3.12 Out-of-sample portfolio evaluation with transaction costs 
% We follow the formulation:
%   w_t ∈ argmax_{w∈W} ( w' μ̂_{t+1} - (γ/2) w' Σ̂_{t+1} w - κ ||w - w_{t-}||_1 ),
%   W = { w ≥ 0, 1'w = 1 }.
% and update wealth with realized costs:
%   W_{t+1} = W_t * (1 - κ||w_t - w_{t-}||_1) * exp( w_t' r_{t+1} ).


W0    = 1e6;
gamma = gammaHat;
kappa = 0.001;                 % 0.1% transaction cost per unit turnover
R_eval    = R_out;
Date_eval = Date_out;
[T_out, d] = size(R_eval);
rebStep     = 3;               % rebalance every n trading days
useLongOnly = true;
useBudget1  = true;
ridge       = 1e-8;
proj_simplex = @(w) project_to_simplex(w);
% Wealth path
W_tc = NaN(T_out+1,1);
W_tc(1) = W0;
% Storage (rebalance dates only)
nReb       = floor((T_out-1)/rebStep) + 1;
Wgt_tc     = NaN(nReb, d);
RebDates   = NaT(nReb,1);
Turnover_tc = NaN(nReb,1);
Cost_tc     = NaN(nReb,1);
% Initialize CCC recursion states using last in-sample day
eps_prev = EPS(end,:).';      % d x 1
h_prev   = SIG2(end,:).';     % d x 1
muVec       = CCC.muVec(:);
omegaFixVec = CCC.omegaFixVec(:);
alphaVec    = CCC.alphaVec(:);
betaVec     = CCC.betaVec(:);
Rhat        = CCC.Rhat;
% Initial portfolio (equal-weight)
w_curr = ones(d,1)/d;         % this will play the role of w_{t-} at the first rebalance
rebIdx = 0;

for t = 1:T_out

    isReb = (t == 1) || (mod(t-1, rebStep) == 0);

    
    % Rebalance (choose w_t given w_{t-} = w_curr)
 
    if isReb
        rebIdx = rebIdx + 1;
        RebDates(rebIdx) = Date_eval(t);

        % 1-step-ahead variance forecast: h_{t|t-1}
        h_fc = omegaFixVec + alphaVec .* (eps_prev.^2) + betaVec .* h_prev;
        h_fc = max(h_fc, 1e-12);

        % Conditional covariance forecast: Σ̂_{t+1} = D R D
        D = diag(sqrt(h_fc));
        Sigma_fc = D * Rhat * D;
        Sigma_fc = 0.5*(Sigma_fc + Sigma_fc.') + ridge*eye(d);

        % Conditional mean forecast 
        mu_fc = muVec;

        % Pre-rebalancing weights w_{t-}
        w_pre = w_curr;

        % Solve TC-penalized Markowitz on simplex:
        % max w' mu - (γ/2) w' Σ w - κ ||w - w_pre||_1
        w_star = solve_markowitz_tc_L1(mu_fc, Sigma_fc, gamma, kappa, w_pre);

        % Realized turnover + realized transaction cost at this rebalance
        turnover = sum(abs(w_star - w_pre));     % ||w_t - w_{t-}||_1
        tc = kappa * turnover;                   % proportional wealth haircut fraction

        % Apply realized cost ONCE as wealth haircut at rebalance
        W_tc(t) = W_tc(t) * (1 - tc);

        % Update current weights AFTER paying costs
        w_curr = w_star;

        % Store
        Wgt_tc(rebIdx,:)    = w_curr.';
        Turnover_tc(rebIdx) = turnover;
        Cost_tc(rebIdx)     = tc;
    end

    
    % Realized portfolio return for day t (using current weights)
    
    r_t = R_eval(t,:).';                    % log-returns at day t
    rp_gross = w_curr' * r_t;               % portfolio log-return

    % Wealth update (NO extra tc subtraction here)
    W_tc(t+1) = W_tc(t) * exp(rp_gross);

    
    % Update CCC states using realized eps_t and h_t
    
    eps_t = r_t - muVec;
    h_t   = omegaFixVec + alphaVec .* (eps_prev.^2) + betaVec .* h_prev;
    h_t   = max(h_t, 1e-12);

    eps_prev = eps_t;
    h_prev   = h_t;
end

% Trim stored arrays
Wgt_tc      = Wgt_tc(1:rebIdx,:);
RebDates    = RebDates(1:rebIdx);
Turnover_tc = Turnover_tc(1:rebIdx);
Cost_tc     = Cost_tc(1:rebIdx);

% Buy-and-hold benchmark same as before
mu_static    = mean(R_in_clean, 1)';     % d x 1
Sigma_static = cov(R_in_clean, 1);       % d x d
Sigma_static = 0.5*(Sigma_static + Sigma_static.') + ridge*eye(d);

w_bh = (1/gamma) * (Sigma_static \ mu_static);

if useBudget1
    s = sum(w_bh);
    if abs(s) < 1e-12
        w_bh = ones(d,1)/d;
    else
        w_bh = w_bh / s;
    end
end
if useLongOnly
    w_bh = proj_simplex(w_bh);
end

W_bh = NaN(T_out+1,1);
W_bh(1) = W0;
for t = 1:T_out
    rp_t = w_bh' * R_eval(t,:).';
    W_bh(t+1) = W_bh(t) * exp(rp_t);
end

% Plot wealth paths
figure;
plot([Date_eval(1); Date_eval], W_tc, 'LineWidth', 1.5); hold on;
plot([Date_eval(1); Date_eval], W_bh, 'LineWidth', 1.5);

if exist('W_dyn','var') && numel(W_dyn) == (T_out+1)
    plot([Date_eval(1); Date_eval], W_dyn, 'LineWidth', 1.5);
    legend('Dynamic CCC (with TC)','Buy&Hold','Dynamic CCC (no TC)','Location','best');
else
    legend('Dynamic CCC (with TC)','Buy&Hold','Location','best');
end

grid on;
title(sprintf('Out-of-sample wealth (rebStep=%d, TC kappa=%.4f)', rebStep, kappa), 'Interpreter','none');
xlabel('Date'); ylabel('Wealth');

%Performance metrics
annFactor = 252;
ret_tc = diff(log(W_tc));
ret_bh = diff(log(W_bh));
mu_tc = mean(ret_tc,'omitnan');
sd_tc = std(ret_tc,'omitnan');
sh_tc = (mu_tc/sd_tc)*sqrt(annFactor);
mu_bh = mean(ret_bh,'omitnan');
sd_bh = std(ret_bh,'omitnan');
sh_bh = (mu_bh/sd_bh)*sqrt(annFactor);
fprintf('\n=== Out-of-sample performance (with transaction costs) ===\n');
fprintf('Dynamic CCC (TC): mean=%.4g, sd=%.4g, Sharpe=%.3f, final wealth=%.2f\n', mu_tc, sd_tc, sh_tc, W_tc(end));
fprintf('Buy&Hold:        mean=%.4g, sd=%.4g, Sharpe=%.3f, final wealth=%.2f\n', mu_bh, sd_bh, sh_bh, W_bh(end));
% Transaction cost reporting
totalTC = sum(Cost_tc,'omitnan');   % sum of proportional haircuts per rebalance (not dollar)
avgTurn = mean(Turnover_tc,'omitnan');
fprintf('Total TC paid (sum kappa*turnover across rebalances) = %.6g\n', totalTC);
fprintf('Average turnover per rebalance = %.4f\n', avgTurn);
W_bh_table = array2table(w_bh.', 'VariableNames', FULLNAMES);
disp('Buy-and-hold weights (static MV):');
disp(W_bh_table);
W_tc_table = array2table(Wgt_tc, 'VariableNames', FULLNAMES);
W_tc_table.Date = RebDates;
W_tc_table = movevars(W_tc_table,'Date','Before',1);
W_tc_table.Turnover = Turnover_tc;
W_tc_table.TC = Cost_tc;
disp('Dynamic weights with transaction costs (rebalance dates):');
disp(W_tc_table);


%% 3.13 Statistical comparison tests: Buy&Hold vs Dynamic CCC (NO TC)
% We compare out-of-sample performance using tests/metrics:
% (1) HAC/Newey-West test of mean return difference
% (2) Block bootstrap test of Sharpe ratio difference
% (3) Tail/path risk comparison: VaR + Max Drawdown (bootstrap)

annFactor  = 252;
alphaLevel = 0.05;

% daily log-returns (NO TC strategy vs buy&hold)
ret_dyn = diff(log(W_dyn));
ret_bh  = diff(log(W_bh));

T = numel(ret_dyn);
if numel(ret_bh) ~= T
    error('ret_dyn and ret_bh length mismatch.')
end
ok = isfinite(ret_dyn) & isfinite(ret_bh);
ret_dyn = ret_dyn(ok);
ret_bh  = ret_bh(ok);
T = numel(ret_dyn);
% Return difference series
d = ret_dyn - ret_bh;
fprintf('\n===== Section 3.13 (NO TC): Tests comparing CCC(no TC) vs Buy&Hold =====\n');
fprintf('Sample size (days): %d\n', T);

% (1) HAC / Newey–West test: mean return difference E[d_t] = 0
% H0: mean(d)=0  vs H1: mean(d) != 0
L = nw_lag_auto(T);  % automatic lag choice
[tstat_d, pval_d, se_d, mean_d] = nw_ttest_mean(d, L);

fprintf('\n(1) HAC mean-difference test (d_t = ret_dyn - ret_BH)\n');
fprintf('  NW lags L = %d\n', L);
fprintf('  mean(d) = %.6g,  NW-SE = %.6g\n', mean_d, se_d);
fprintf('  t-stat = %.4f,  p-value = %.4g\n', tstat_d, pval_d);

% (2) Block bootstrap test: Sharpe ratio difference
% H0: SR_dyn - SR_BH = 0 (two-sided)
B = max(5, round(T^(1/3)));
nBoot = 50000;
rng(1);

[srdiff_hat, pval_sr, CI_sr] = boot_sharpe_diff_block(ret_dyn, ret_bh, annFactor, nBoot, B);

fprintf('\n(2) Block-bootstrap Sharpe difference test\n');
fprintf('  Block length B = %d, boot reps = %d\n', B, nBoot);
fprintf('  SR diff (dyn - BH) = %.6g\n', srdiff_hat);
fprintf('  95%% bootstrap CI = [%.6g, %.6g]\n', CI_sr(1), CI_sr(2));
fprintf('  bootstrap p-value (two-sided) = %.4g\n', pval_sr);

% (3) Tail/path risk comparison: VaR(5%) and Max Drawdown (bootstrap)
q = 0.05;
[Vdiff_hat, pval_V, CI_V, MDDdiff_hat, pval_MDD, CI_MDD] = boot_VaR_MDD_diff_block(ret_dyn, ret_bh, q, nBoot, B);
fprintf('\n(4) Bootstrap tail/path risk differences (dyn - BH)\n');
fprintf('  VaR(%.0f%%) diff = %.6g, 95%% CI=[%.6g, %.6g], p=%.4g\n', 100*q, Vdiff_hat, CI_V(1), CI_V(2), pval_V);
fprintf('  MDD diff        = %.6g, 95%% CI=[%.6g, %.6g], p=%.4g\n', MDDdiff_hat, CI_MDD(1), CI_MDD(2), pval_MDD);
sr_dyn = (mean(ret_dyn)/std(ret_dyn))*sqrt(annFactor);
sr_bh  = (mean(ret_bh)/std(ret_bh))*sqrt(annFactor);
VaR_dyn = -quantile(ret_dyn, q);
VaR_bh  = -quantile(ret_bh, q);
MDD_dyn = max_drawdown_from_returns(ret_dyn);
MDD_bh  = max_drawdown_from_returns(ret_bh);

Summary_noTC = table( ...
    [mean(ret_dyn); mean(ret_bh)], ...
    [std(ret_dyn);  std(ret_bh)], ...
    [sr_dyn;        sr_bh], ...
    [VaR_dyn;       VaR_bh], ...
    [MDD_dyn;       MDD_bh], ...
    'VariableNames', {'MeanLogRet','StdLogRet','Sharpe','VaR5','MaxDrawdown'}, ...
    'RowNames', {'CCC_no_TC','BuyHold'} );

disp(' ');
disp('Point-estimate metrics (out-of-sample, NO TC):');
disp(Summary_noTC);
fprintf('\n===== End of Section 3.13 (NO TC) =====\n');


%% 3.14 Statistical comparison tests: Buy&Hold vs Dynamic CCC (with TC)
% We compare out-of-sample performance using tests/metrics same as 3.13
annFactor = 252;
alphaLevel = 0.05;
% daily log-returns 
ret_tc = diff(log(W_tc));        
ret_bh = diff(log(W_bh));       

T = numel(ret_tc);
if numel(ret_bh) ~= T
    error('ret_tc and ret_bh length mismatch.')
end
ok = isfinite(ret_tc) & isfinite(ret_bh);
ret_tc = ret_tc(ok);
ret_bh = ret_bh(ok);
T = numel(ret_tc);
% Return difference series
d = ret_tc - ret_bh;
fprintf('\n===== Section 3.14: Tests comparing CCC(with TC) vs Buy&Hold =====\n');
fprintf('Sample size (days): %d\n', T);

% (1) HAC / Newey–West test: mean return difference E[d_t] = 0
% H0: mean(d)=0  vs H1: mean(d) != 0
L = nw_lag_auto(T);  % automatic lag choice
[tstat_d, pval_d, se_d, mean_d] = nw_ttest_mean(d, L);
fprintf('\n(1) HAC mean-difference test (d_t = ret_TC - ret_BH)\n');
fprintf('  NW lags L = %d\n', L);
fprintf('  mean(d) = %.6g,  NW-SE = %.6g\n', mean_d, se_d);
fprintf('  t-stat = %.4f,  p-value = %.4g\n', tstat_d, pval_d);
% (2) Block bootstrap test: Sharpe ratio difference
% H0: SR_TC - SR_BH = 0 (two-sided)
% We use moving-block bootstrap to respect dependence in returns.
B = max(5, round(T^(1/3)));   
nBoot = 50000;                 % number of draws
rng(1);                        % reproducible
[srdiff_hat, pval_sr, CI_sr] = boot_sharpe_diff_block(ret_tc, ret_bh, annFactor, nBoot, B);
fprintf('\n(2) Block-bootstrap Sharpe difference test\n');
fprintf('  Block length B = %d, boot reps = %d\n', B, nBoot);
fprintf('  SR diff (TC - BH) = %.6g\n', srdiff_hat);
fprintf('  95%% bootstrap CI = [%.6g, %.6g]\n', CI_sr(1), CI_sr(2));
fprintf('  bootstrap p-value (two-sided) = %.4g\n', pval_sr);

% (3) Tail/path risk comparison: VaR(5%) and Max Drawdown (bootstrap)
% We compare:
%   VaR_5% of daily returns  (loss quantile)
%   Max Drawdown of wealth path
% We bootstrap the DIFFERENCE (TC - BH) using moving-block bootstrap.
q = 0.05;
[Vdiff_hat, pval_V, CI_V, MDDdiff_hat, pval_MDD, CI_MDD] = boot_VaR_MDD_diff_block( ret_tc, ret_bh, q, nBoot, B);
fprintf('\n(4) Bootstrap tail/path risk differences (TC - BH)\n');
fprintf('  VaR(%.0f%%) diff = %.6g, 95%% CI=[%.6g, %.6g], p=%.4g\n',  100*q, Vdiff_hat, CI_V(1), CI_V(2), pval_V);
fprintf('  MDD diff        = %.6g, 95%% CI=[%.6g, %.6g], p=%.4g\n',  MDDdiff_hat, CI_MDD(1), CI_MDD(2), pval_MDD);
sr_tc = (mean(ret_tc)/std(ret_tc))*sqrt(annFactor);
sr_bh = (mean(ret_bh)/std(ret_bh))*sqrt(annFactor);
VaR_tc = -quantile(ret_tc, q);
VaR_bh = -quantile(ret_bh, q);
MDD_tc = max_drawdown_from_returns(ret_tc);
MDD_bh = max_drawdown_from_returns(ret_bh);
Summary = table( ...
    [mean(ret_tc); mean(ret_bh)], ...
    [std(ret_tc);  std(ret_bh)], ...
    [sr_tc;        sr_bh], ...
    [VaR_tc;       VaR_bh], ...
    [MDD_tc;       MDD_bh], ...
    'VariableNames', {'MeanLogRet','StdLogRet','Sharpe','VaR5','MaxDrawdown'}, ...
    'RowNames', {'CCC_with_TC','BuyHold'} );
disp(' ');
disp('Point-estimate metrics (out-of-sample):');
disp(Summary);
fprintf('\n===== End of Section 3.14 =====\n');

%% Functions

function L = nw_lag_auto(T)
% Simple automatic lag choice for Newey–West
L = floor(4*(T/100)^(2/9));
L = max(L, 1);
end

function [tstat, pval, se, mu] = nw_ttest_mean(x, L)
% Newey–West HAC t-test for H0: E[x]=0
x = x(:);
T = numel(x);
mu = mean(x);

u = x - mu;
gamma0 = mean(u.^2);

% autocovariances
g = zeros(L,1);
for ell = 1:L
    g(ell) = mean(u(1+ell:end).*u(1:end-ell));
end

% Bartlett weights
varHAC = gamma0;
for ell = 1:L
    w = 1 - ell/(L+1);
    varHAC = varHAC + 2*w*g(ell);
end

% SE of mean
se = sqrt(varHAC / T);

tstat = mu / se;

% Use t approx 
df = T-1;
pval = 2*(1 - tcdf(abs(tstat), df));
end

function [srdiff_hat, pval, CI] = boot_sharpe_diff_block(r1, r2, annFactor, nBoot, B)
% Moving-block bootstrap for Sharpe ratio difference SR1 - SR2
r1 = r1(:); r2 = r2(:);
T = numel(r1);

sr = @(r) (mean(r)/std(r))*sqrt(annFactor);

srdiff_hat = sr(r1) - sr(r2);

bootStats = zeros(nBoot,1);
for b = 1:nBoot
    idx = moving_block_indices(T, B);
    rb1 = r1(idx);
    rb2 = r2(idx);
    bootStats(b) = sr(rb1) - sr(rb2);
end

% two-sided p-value via centered bootstrap distribution
centered = bootStats - mean(bootStats);
pval = mean(abs(centered) >= abs(srdiff_hat - mean(bootStats)));

% percentile CI
CI = quantile(bootStats, [0.025, 0.975]);
end

function [Vdiff_hat, pval_V, CI_V, MDDdiff_hat, pval_MDD, CI_MDD] = boot_VaR_MDD_diff_block(r1, r2, q, nBoot, B)
% Bootstraps difference in VaR(q) and MaxDrawdown using block bootstrap
r1 = r1(:); r2 = r2(:);
T = numel(r1);

VaR = @(r) -quantile(r, q);                 % positive loss number
MDD = @(r) max_drawdown_from_returns(r);    % positive number

Vdiff_hat   = VaR(r1) - VaR(r2);
MDDdiff_hat = MDD(r1) - MDD(r2);

bootV   = zeros(nBoot,1);
bootMDD = zeros(nBoot,1);

for b = 1:nBoot
    idx = moving_block_indices(T, B);
    rb1 = r1(idx);
    rb2 = r2(idx);

    bootV(b)   = VaR(rb1) - VaR(rb2);
    bootMDD(b) = MDD(rb1) - MDD(rb2);
end

% two-sided p-values (centered)
cV = bootV - mean(bootV);
pval_V = mean(abs(cV) >= abs(Vdiff_hat - mean(bootV)));

cM = bootMDD - mean(bootMDD);
pval_MDD = mean(abs(cM) >= abs(MDDdiff_hat - mean(bootMDD)));

CI_V   = quantile(bootV,   [0.025, 0.975]);
CI_MDD = quantile(bootMDD, [0.025, 0.975]);
end

function idx = moving_block_indices(T, B)
% Create indices 1..T using moving-block bootstrap of block length B
nBlocks = ceil(T / B);
start = randi(T - B + 1, nBlocks, 1);
idx = zeros(nBlocks*B,1);
pos = 1;
for k = 1:nBlocks
    idx(pos:pos+B-1) = (start(k):(start(k)+B-1)).';
    pos = pos + B;
end
idx = idx(1:T);
end

function mdd = max_drawdown_from_returns(r)
% Max drawdown from log returns r (T x 1)
% Wealth path normalized to 1
W = exp(cumsum(r));
peak = cummax(W);
dd = 1 - W ./ peak;     % drawdown in [0,1)
mdd = max(dd);
end

function [b, se_b, t_b, p_b] = hac_ols(X, y, L)
% OLS with Newey–West HAC standard errors for coefficients
% y = X*b + u
y = y(:);
[T, k] = size(X);

b = X \ y;
u = y - X*b;

% Compute HAC covariance of b:
% V = (X'X)^(-1) * (X' S X) * (X'X)^(-1), where S is HAC of u
XXi = inv(X'*X);

% Build meat: X' * Omega * X via sums of lagged products
S = zeros(k,k);

% lag 0
for t = 1:T
    xt = X(t,:).';
    S = S + (u(t)^2) * (xt*xt.');
end

% lags 1..L with Bartlett weights
for ell = 1:L
    w = 1 - ell/(L+1);
    for t = (ell+1):T
        xt  = X(t,:).';
        xtL = X(t-ell,:).';
        S = S + w * (u(t)*u(t-ell)) * (xt*xtL.' + xtL*xt.');
    end
end

S = S / T;                 % scale
V = XXi * S * XXi / T;      % consistent for Var(b)

se_b = sqrt(diag(V));
t_b = b ./ se_b;

df = T - k;
p_b = 2*(1 - tcdf(abs(t_b), df));
end

function w = project_to_simplex(v)
% Projects v onto {w >= 0, sum(w)=1}.
v = v(:);
n = numel(v);
u = sort(v,'descend');
cssv = cumsum(u) - 1;
rho = find(u - cssv./(1:n)' > 0, 1, 'last');
theta = cssv(rho)/rho;
w = max(v - theta, 0);
end

function w = solve_markowitz_tc_L1(mu, Sigma, gamma, kappa, w_prev)
% Solve:
%   max_{w in simplex}  w'*mu - (gamma/2) w'*Sigma*w - kappa*||w - w_prev||_1
% Equivalently:
%   min_{w in simplex}  (gamma/2) w'*Sigma*w - mu'*w + kappa*||w - w_prev||_1

    mu = mu(:);
    d  = numel(mu);

    Sigma = 0.5*(Sigma + Sigma');  % symmetrize

    % Lipschitz constant of grad(gamma/2 w'Sigma w - mu'w) is gamma*lambda_max(Sigma)
    lamMax = max(eig(Sigma));
    if ~isfinite(lamMax) || lamMax <= 0
        lamMax = 1;
    end
    step = 1 / (gamma*lamMax + 1e-12);

    maxIter = 500;
    tol     = 1e-10;

    w = w_prev(:);
    if any(~isfinite(w)) || abs(sum(w)-1) > 1e-6 || any(w<0)
        w = ones(d,1)/d;
    end

    for it = 1:maxIter
        grad = gamma*(Sigma*w) - mu;     % gradient of smooth part

        x = w - step*grad;              % gradient step

        % prox for kappa*||w-w_prev||_1:
        x = w_prev + soft_threshold(x - w_prev, step*kappa);

        % project onto simplex: w>=0, sum w=1
        w_new = project_to_simplex(x);

        if norm(w_new - w, 1) < tol
            w = w_new;
            break
        end
        w = w_new;
    end
end

function y = soft_threshold(x, tau)
% elementwise soft-thresholding
    y = sign(x) .* max(abs(x) - tau, 0);
end

function l = lnl_dcc_gauss_constr(u, Z)
    % u -> (a,b) constrained: a>0, b>0, a+b<1
    [a,b] = dcc_map(u);

    % If constraints violated numerically, give huge penalty
    if ~(isfinite(a) && isfinite(b)) || a<=0 || b<=0 || (a+b)>=1
        T = size(Z,1);
        l = -1e10*ones(T,1);
        return
    end

    % Build DCC loglik contributions (corr part only)
    [Rseq,~] = dcc_filter(Z, a, b);
    T = size(Z,1); d = size(Z,2);
    l = zeros(T,1);

    for t = 1:T
        Rt = Rseq(:,:,t);

        
        Rt = 0.5*(Rt+Rt');
        % small jitter if near-singular
        [~,p] = chol(Rt);
        if p~=0
            Rt = Rt + 1e-10*eye(d);
        end

        % loglik corr-part: -0.5*(log|R| + z'R^{-1}z)
        zt = Z(t,:)';
        [L,p] = chol(Rt,'lower');
        if p~=0
            l(t) = -1e10;
            continue
        end
        
        y = L \ zt;
        quad = y' * y;                     
        logdetR = 2*sum(log(diag(L)));      
        l(t) = -0.5*(logdetR + quad);
    end
end

function [a,b] = dcc_map(u)
    % u1 -> a>0 via exp
    a = exp(u(1));

    % u2 -> b in (0,1) via logistic
    b = 1/(1+exp(-u(2)));

    % enforce a+b<1 by shrinking b 
    if a + b >= 0.999999
        b = 0.999999 - a;
    end
end

function [Rseq,Qseq] = dcc_filter(Z, a, b)
    
    [T,d] = size(Z);

    % Unconditional correlation proxy
    Qbar = cov(Z,1);      
    Qbar = 0.5*(Qbar+Qbar');

    Qseq = zeros(d,d,T);
    Rseq = zeros(d,d,T);

    Qseq(:,:,1) = Qbar;

    for t = 2:T
        zprev = Z(t-1,:)';
        Qseq(:,:,t) = (1-a-b)*Qbar + a*(zprev*zprev') + b*Qseq(:,:,t-1);
        Qseq(:,:,t) = 0.5*(Qseq(:,:,t)+Qseq(:,:,t)');  % sym
    end

    for t = 1:T
        Qt = Qseq(:,:,t);
        dQt = sqrt(max(diag(Qt),1e-12));
        Dinv = diag(1./dQt);
        Rt = Dinv*Qt*Dinv;
        Rt = 0.5*(Rt+Rt');
        Rt(1:d+1:end) = 1;  % force exact ones on diagonal
        Rseq(:,:,t) = Rt;
    end
end

function l = lnl_ccc_corr_only(Z, Rhat)
    % corr-part only under CCC
    [T,d] = size(Z);
    l = zeros(T,1);

    Rhat = 0.5*(Rhat+Rhat');
    [L,p] = chol(Rhat,'lower');
    if p~=0
        % jitter if needed
        Rhat = Rhat + 1e-10*eye(d);
        L = chol(Rhat,'lower');
    end
    logdetR = 2*sum(log(diag(L)));

    for t=1:T
        zt = Z(t,:)';
        y = L \ zt;
        quad = y'*y;
        l(t) = -0.5*(logdetR + quad);
    end
end

function l = ccc_loglik_gauss(EPS, SIG2, R)

    [T,d] = size(EPS);
    l = NaN(T,1);

    % Precompute
    logdetR = log(det(R));
    Rinv = inv(R);

    for t = 1:T
        sig2_t = SIG2(t,:).';
        eps_t  = EPS(t,:).';

        if any(~isfinite(sig2_t)) || any(sig2_t<=0) || any(~isfinite(eps_t))
            continue
        end

        % log det(H_t) = log det(R) + sum log(sig2_i,t)
        logdetH = logdetR + sum(log(sig2_t));

        % e' H^{-1} e = z' R^{-1} z, where z_i = e_i / sqrt(sig2_i)
        z_t = eps_t ./ sqrt(sig2_t);
        quad = z_t' * Rinv * z_t;

        l(t) = -0.5*( d*log(2*pi) + logdetH + quad );
    end
end

function [A_pd, jitter] = makePD(A)
% Minimal PD repair: add diagonal jitter until chol works
    A = 0.5*(A + A.');
    jitter = 0;
    % if already PD
    [~,p] = chol(A);
    if p == 0
        A_pd = A;
        return
    end

    jitter = 1e-10;
    for k = 1:15
        A_try = A + jitter*eye(size(A));
        [~,p] = chol(A_try);
        if p == 0
            A_pd = A_try;
            return
        end
        jitter = jitter*10;
    end

  
    A_pd = A + jitter*eye(size(A));
end

function acf = local_acf(x, L)
    x = x(:);
    x = x - mean(x);
    n = length(x);
    denom = sum(x.^2);
    acf = zeros(L+1,1);
    for k = 0:L
        acf(k+1) = (x(1:n-k)'*x(1+k:n)) / max(denom, eps);
    end
end

function l = lnl_garch_spec_fixomega(par, r, omegaFix)
mu    = par(1);
alpha = par(2);
beta  = par(3);
cTail = par(4);
fTail = par(5);
T = length(r);
l = NaN(T,1);
if ~(isfinite(omegaFix)) || (omegaFix <= 0) || (alpha < 0) || (beta < 0) || (alpha + beta >= 1)
    return
end
if (cTail < 0) || (cTail >= 1) || (fTail <= 0)
    return
end
eps = r - mu;
h = zeros(T,1);
h(1) = var(r);
for t = 2:T
    h(t) = omegaFix + alpha*eps(t-1)^2 + beta*h(t-1);
    if ~(isfinite(h(t))) || (h(t) <= 0)
        l = NaN(T,1);
        return
    end
end
s_t   = sqrt(h);
parHF = [cTail; fTail];
l = lnl_spec(parHF, s_t, eps);
end

function l = lnl_garch_spec_fixomega_BTC_stationary(par, r, omegaFix, epsS)
mu = par(1);
a  = par(2);
b  = par(3);
u  = par(4);
v  = par(5);

T = length(r);
l = NaN(T,1);

if ~(isfinite(omegaFix)) || (omegaFix <= 0)
    return
end

ea = exp(a);
eb = exp(b);
den = 1 + ea + eb;
alpha = (ea/den) * (1 - epsS);
beta  = (eb/den) * (1 - epsS);

cTail = 0.999 / (1 + exp(-u));
fTail = exp(v);

eps = r - mu;

h = zeros(T,1);
h(1) = var(r);

for t = 2:T
    h(t) = omegaFix + alpha*eps(t-1)^2 + beta*h(t-1);
    if ~(isfinite(h(t))) || (h(t) <= 0)
        l = NaN(T,1);
        return
    end
end

s_t   = sqrt(h);
parHF = [cTail; fTail];
l = lnl_spec(parHF, s_t, eps);
end

function [alpha, beta, cTail, fTail] = btc_map(thHat, epsS)
a = thHat(2);
b = thHat(3);
u = thHat(4);
v = thHat(5);

ea = exp(a);
eb = exp(b);
den = 1 + ea + eb;
alpha = (ea/den) * (1 - epsS);
beta  = (eb/den) * (1 - epsS);

cTail = 0.999 / (1 + exp(-u));
fTail = exp(v);
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

function T = waldCI_table(thetaHat, VCV, rowNames)

    se = sqrt(diag(VCV));
    CI_low  = thetaHat - 1.96*se;
    CI_high = thetaHat + 1.96*se;
    T = table(thetaHat(:), se(:), CI_low(:), CI_high(:), ...
        'VariableNames', {'Estimate','SE','CI_low','CI_high'}, ...
        'RowNames', rowNames);
end



