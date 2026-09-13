clear all;
clc;
load coal_mine_disasters.mat
%%1.2 and 3   4 5 
format short g

% Must adjust and tune the parameters below:
d = 6;  % Set d = 1 2 3 4 5 6
lambda = 0.02 * ones(1, d);   % initial intensities 

Psi = 1;    %fixed hyperparameter for theta's prior
theta = 1/Psi;                  % hyperparameter for lambda's prior
% MCMC settings
N_iter = 12000;   % total number of iterations
burn_in  = 2000;  % burn-in period to discard
rho = 0.034;       % tuning parameter for MH proposals on breakpoints




% First observe cumulative plot to spot potential rate of changes. 
% [t_sorted, idx] = sort(tau); 
% n = length(t_sorted);
% figure;
% stairs(t_sorted, 1:n, 'LineWidth', 2);
% xlabel('Year');
% ylabel('Cumulative Number of Disasters');
% title('Cumulative Plot of Coal Mine Disasters');
% grid on;



total_MH =0 ;
accepted_MH = 0;
t1 = 1658;
t_end = 1980;
% Initialize MH
if d == 1
    t = [t1, t_end];
else
    t = linspace(t1, t_end, d+1);
end
% Initialize MCMC
samples_t = zeros(N_iter, length(t));   
samples_lambda = zeros(N_iter, d);
samples_theta = zeros(N_iter, 1);

% count disasters in each interval
countDisasters = @(tau, tvec) arrayfun(@(i) sum(tau >= tvec(i) & tau < tvec(i+1)), 1:(length(tvec)-1));

%%MCMC Loop
for m = 1:N_iter
    %Gibbs 
    n_i = countDisasters(tau, t);  
    for i = 1:d
        shape_post = 2 + n_i(i);
        rate_post  = theta + (t(i+1) - t(i));
        lambda(i)  = gamrnd(shape_post, 1/rate_post);
    end
   %update theta 
    shape_theta = 2 + 2*d;
    rate_theta  = Psi + sum(lambda);
    theta = gamrnd(shape_theta, 1/rate_theta);
    % update breakpoints MH
    if d > 1
        for i = 2:d
            total_MH = total_MH + 1;
            t_old = t(i);
            R = rho * (t(i+1) - t(i-1));
            t_new = t_old + (2*rand - 1)*R;
            
            % Enforce ordering: t(i-1) < t_new < t(i+1)
            if (t_new <= t(i-1)) || (t_new >= t(i+1))
                continue;   
            end
             % Create new breakpoint vector with the proposed t(i)
            t_new_vec = t;
            t_new_vec(i) = t_new;
            delta_old = [ t(i) - t(i-1), t(i+1) - t(i) ];
            delta_new = [ t_new_vec(i) - t_new_vec(i-1), t_new_vec(i+1) - t_new_vec(i) ];
            n_old = countDisasters(tau, t);
            n_new = countDisasters(tau, t_new_vec);
            logLike_old = n_old(i-1)*log(lambda(i-1)) - lambda(i-1)*delta_old(1) + ...
                          n_old(i)*log(lambda(i))   - lambda(i)*delta_old(2);
            logLike_new = n_new(i-1)*log(lambda(i-1)) - lambda(i-1)*delta_new(1) + ...
                          n_new(i)*log(lambda(i))   - lambda(i)*delta_new(2);
            logPrior_old = log(delta_old(1)) + log(delta_old(2));
            logPrior_new = log(delta_new(1)) + log(delta_new(2));
            log_post_old = logLike_old + logPrior_old;
            log_post_new = logLike_new + logPrior_new;
            
            % MH acceptance probability 
        
            alpha = exp(log_post_new - log_post_old);
            if rand < alpha
                t(i) = t_new;
                accepted_MH = accepted_MH + 1;
            end
        end
    end
    
    samples_t(m, :) = t;
    samples_lambda(m, :) = lambda;
    samples_theta(m) = theta;
end

% Discard Burn-in
idx_keep = (burn_in+1):N_iter;
samples_t = samples_t(idx_keep, :);
samples_lambda = samples_lambda(idx_keep, :);
samples_theta = samples_theta(idx_keep);
% Compute posterior means
mean_t = mean(samples_t, 1);
mean_lambda = mean(samples_lambda, 1);
mean_theta = mean(samples_theta);
% Calculate MH acceptance rate for breakpoint proposals
acceptance_rate = accepted_MH / total_MH;
fprintf('MH acceptance rate for breakpoints: %.2f%%\n', acceptance_rate*100);

fprintf('Posterior means of breakpoints:\n');
disp(mean_t);
fprintf('Posterior means of lambda:\n');
disp(mean_lambda);
fprintf('Posterior mean of theta: %.4f\n', mean_theta);
%Plotting 
if d > 1
    figure;
    colors = lines(d-1); % Generate d-1 distinct colors
    hold on;
    for j = 2:d  % Plot breakpoints t(2) to t(d)
        plot(samples_t(:, j), 'Color', colors(j-1, :), 'LineWidth', 2);
    end
    xlabel('Iteration (post burn-in)');
    ylabel('Breakpoint Estimate');
    title('Trace Plots for All Unknown Breakpoints');
    legend(arrayfun(@(x) sprintf('t_{%d}', x), 2:d, 'UniformOutput', false));
    grid on;
    hold off;
end

%% part 2.
clear
clc
load atlantic.txt

% F inverse
f_inv = @(u, mu, beta) mu - beta * log(-log(u));

%Parameter estimates
[beta_est,mu_est] = est_gumbel(atlantic);
disp("Mu estimate: " + mu_est);
disp("Beta estimate: " + beta_est);

%initialize
n = length(atlantic);
B = 1000;
boot_beta = zeros(B,1);
boot_mu = zeros(B,1);

% bootstrap
for b = 1:B
    U = rand(n,1);
    boot = f_inv(U, mu_est, beta_est);
    [beta_boot, mu_boot]  = est_gumbel(boot);
    boot_beta(b) = beta_boot;
    boot_mu(b) = mu_boot;
end
delta_beta = sort(beta_est - boot_beta);
delta_mu = sort(mu_est - boot_mu);

% create the 99% confidence interval
alpha = 0.01;
Beta_L = beta_est - delta_beta(ceil((1 - alpha / 2) * B));
Beta_U= beta_est - delta_beta(ceil(alpha * B / 2));
Mu_L = mu_est - delta_mu(ceil((1 - alpha / 2) * B));
Mu_U= mu_est - delta_mu(ceil(alpha * B / 2));

disp("99% Confidence interval for mu: (" + Mu_L + ", " + Mu_U + ").");
disp("99% Confidence interval for beta: (" + Beta_L + ", " + Beta_U + ").");
%% 
T = 3 * 14 * 100;
boot_w = zeros(B,1);
wave_est = f_inv(1 - 1 / T, mu_est, beta_est);
for b = 1:B
      boot_w(b,1) = f_inv(1 - 1 / T, boot_mu(b), boot_beta(b));
end

delta_wave = sort(boot_w - wave_est);
wave_U = wave_est - delta_wave(ceil(alpha * B/2));

disp("Wave-height estimate: " + wave_est)
disp("Wave-height 99% upper bound: " + wave_U)



