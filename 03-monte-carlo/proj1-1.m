
clear all;
load powercurve_V164.mat
month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
lambda = [10.6 9.7 9.2 8.0 7.8 8.1 7.8 8.1 9.1 9.9 10.6 10.6];
k = [2.0 2.0 2.0 1.9 1.9 1.9 1.9 1.9 2.0 1.9 2.0 2.0];
N = 100000;
steps = 100;
sqN = sqrt(N);
se = 1.96;
%% 2.a -- Standard Monte Carlo
disp("------------Standard Monte Carlo------------")
tau = zeros(12, N/steps);
sd = zeros(12, N/steps);
LB = zeros(12, N/steps);
UB = zeros(12, N/steps);
width = zeros(1,12);
for i = 1:12
    q = 1; 
    for j = 100:steps:N
        number = wblrnd(lambda(i), k(i), j, 1);
        power = P(number);
        tau(i,q) = mean(power);
        sd(i,q) = std(power);
        LB(i,q) = tau(i,q) - (se * sd(i,q) / sqrt(j));
        UB(i,q) = tau(i,q) + (se * sd(i,q) / sqrt(j));
        width(1,i) = UB(i,end) - LB(i,end);
        q = q + 1;
    end

    disp(month(i) + ": LB = " + LB(i,end) + ", UB = " + UB(i,end) + "; Width = " + width(1,i));
end
%% 2.a -- Truncated version
disp("------------Truncated version------------");
a = 3.5;
b = 25;
tau_T = zeros(12, N/steps);
sd_T = zeros(12, N/steps);
LB_T = zeros(12, N/steps);
UB_T = zeros(12, N/steps);
width_T = zeros(1,12);
for i = 1:12
    q = 1;
    for j = 100:steps:N
        u = rand(j,1);
        Fxa = wblcdf(a, lambda(i), k(i));
        Fxb = wblcdf(b, lambda(i), k(i));
        temp = Fxa + u * (Fxb - Fxa);
        speed = wblinv(temp, lambda(i), k(i));
        power = P(speed) * (Fxb - Fxa);
        tau_T(i,q) = mean(power);
        sd_T(i,q) = std(power);
        LB_T(i,q) = tau_T(i,q) - (se * sd_T(i,q) / sqrt(j));
        UB_T(i,q) = tau_T(i,q) + (se * sd_T(i,q) / sqrt(j));
        width_T(1,i) = UB_T(i,end) - LB_T(i,end);
        q = q + 1;
    end
    disp(month(i) + ": LB = " + LB_T(i,end) + ", UB = " + UB_T(i,end) + "; Width = " + width_T(1,i));
end
%% 2.a -- comparison
figure(1)
x = 100:steps:N;
y = mean(tau(1,:));
const = @(x)(y).*x.^(0);
plot(x,const(x),'green');
hold on
plot(x, LB(1,:),'--','color','blue');
plot(x, UB(1,:),'--','color','blue');
plot(x, LB_T(1,:),'--','color','red');
plot(x, UB_T(1,:),'--','color','red');
%% 2.b
disp("------------Control Variate------------");
tau_CV = zeros(12, N/steps);
sd_CV = zeros(12, N/steps);
LB_CV = zeros(12, N/steps);
UB_CV = zeros(12, N/steps);
width_CV = zeros(1,12);
for i = 1:12
    q = 1;
    for j = 100:steps:N
        X = wblrnd(lambda(i), k(i), j, 1);
        Y = X;
        c = cov(P(X), Y);
        beta = -c(1,2)./var(Y);
        m = gamma(1+1/k(i))*lambda(i);
        Z = P(X) + beta * (Y - m);
        tau_CV(i,q) = mean(Z);
        sd_CV(i,q) = std(Z);
        LB_CV(i,q) = tau_CV(i,q) - (se * sd_CV(i,q) / sqrt(j));
        UB_CV(i,q) = tau_CV(i,q) + (se * sd_CV(i,q) / sqrt(j));
        width_CV(1,i) = UB_CV(i,end) - LB_CV(i,end);
        q = q + 1;
    end
        disp(month(i) + ": LB = " + LB_CV(i,end) + ", UB = " + UB_CV(i,end) + "; Width = " + width_CV(1,i));
end
%% 2.c
disp("------------Important Sampling------------");

% parameters chosen by hand (by comparisons in plots)
sigma = [4.25, 4.1, 4.05, 3.75, 3.6, 3.8, 3.62, 3.8, 3.8, 4.1, 4.2, 4.2]; 
mu = [12, 11.5, 11, 10.5, 10.5, 10.5, 10.5, 10.5, 11, 12, 12, 12]; 

% initialization
tau_is = zeros(12, 1);       
sd_is = zeros(12, 1);        
LB_is = zeros(12, 1);        
UB_is = zeros(12, 1);        
width_is = zeros(12, 1); 

for i = 1:12
    fx = @(x) wblpdf(x, lambda(i), k(i)); % target density
    gx = @(x) normpdf(x, mu(i), sigma(i)); % instrumental density
    IS = @(x) (fx(x) .* P(x)') ./ gx(x); 
    
    %sampling
    X = sigma(i) * randn(1, N) + mu(i); 
    Z = IS(X); 
    
    % confidence intervals
    tau_is(i) = mean(Z);
    sd_is(i) = std(Z);
    LB_is(i) = tau_is(i) - (se * sd_is(i) / sqrt(N));
    UB_is(i) = tau_is(i) + (se * sd_is(i) / sqrt(N));
    width_is(i) = UB_is(i) - LB_is(i);
    
    disp(month(i) + ": I = (" + LB_is(i) + ", " + UB_is(i) + "); Width = " + width_is(i));
end

%% 2.d
disp("------------Antithetic Sampling------------");
tau_AS = zeros(12, N/steps);
sd_AS = zeros(12, N/steps);
LB_AS = zeros(12, N/steps);
UB_AS = zeros(12, N/steps);
width_AS = zeros(1,12);

for i = 1:12
    q = 1;
    for j = 100:steps:N
        u = rand(j/2, 1); % Half the sample size
        V = P(wblinv(u, lambda(i), k(i)));
        VTilde = P(wblinv(1 - u, lambda(i), k(i)));
        W = (V + VTilde) ./ 2;
        tau_AS(i,q) = mean(W);
        sd_AS(i,q) = std(W);
        LB_AS(i,q) = tau_AS(i,q) - (se * sd_AS(i,q) / sqrt(j));
        UB_AS(i,q) = tau_AS(i,q) + (se * sd_AS(i,q) / sqrt(j));
        width_AS(1,i) = UB_AS(i,end) - LB_AS(i,end);
        q = q + 1;
    end
        disp(month(i) + ": LB = " + LB_AS(i,end) + ", UB = " + UB_AS(i,end) + "; Width = " + width_AS(1,i));
end
%% 2.e
disp("------------Probability the Turbine Delivers Power------------");
for i = 1:12
    v = wblrnd(lambda(i), k(i), N, 1);
    probP = mean(P(v) > 0);         %probability of power
    disp([month(i), ": Probability of power = ", num2str(probP)]);
end
%% 2.f
disp("------------Average Ration------------");
d = 164;
rho = 1.225;

for i = 1:12
    v = wblrnd(lambda(i), k(i), N, 1);
    power = P(v);
    powerAct = mean(power);
    powerTot = 0.5 * rho * pi * 0.25 * d^2 * gamma(1 + 3 / k(i))*lambda(i)^3;
    powerRat = powerAct / powerTot;
    sd = std(power / powerTot);
    LB = powerRat - (se * sd / sqN);
    UB = powerRat + (se * sd / sqN);
    width = UB - LB;
    disp(month(i) + ": LB = " + LB + ", UB = " + UB + "; Width = " + width)
end
%% 2.e
disp("------------Capacity Factor & Availability Factor------------");
capacityF = zeros(1,12);
avaliaF = zeros(1,12);
for i = 1:12
    v = wblrnd(lambda(i), k(i), N, 1);
    power = P(v);
    capacityF(1,i) = mean(power) / 9.5e6;
    avaliaF(1,i) = nnz(power) / numel(power);
end
CF = sum(capacityF)/12;
AF = sum(avaliaF)/12;
disp("Average Capacity Factor: " + CF)
disp("Average Availability Factor: " + AF)

%% 3(a): 
clear; clc; close all;
load powercurve_V164.mat  % Loads the power curve function P(v) in Watts

%Plotting the Functions
k = 1.96;
lambda = 9.13;
x = 0:0.5:30;
fx = @(x) wblpdf(x, lambda, k);
phi = @(x) P(x); 
%weighted function: φ(x)*f(x)
temp = phi(x) .* fx(x)';  % Transpose fx(x) to ensure proper element-wise multiplication
figure;
plot(x, temp, 'b-', 'LineWidth', 2);
hold on;

% instrumental density g(x)
mu_g = 11;
sigma_g = 4.4;
gx = @(x) normpdf(x, mu_g, sigma_g);

% Scale g(x) for better observation
g = gx(x) * 5000000;
plot(x, g, 'r--', 'LineWidth', 2);
legend('\phi(x)*f(x)', 'g(x)', 'Location', 'best');
xlabel('Wind Speed (m/s)');
ylabel('Value');
title('Comparison of \phi(x)*f(x) and Instrumental g(x)');
grid on;
hold off;
IS = @(x) (fx(x) .* P(x)') ./ gx(x);
N = 10000;
v1 = sigma_g .* randn(1, N) + mu_g;
v2 = sigma_g .* randn(1, N) + mu_g;
z1 = IS(v1);
z2 = IS(v2);
tau1 = mean(z1);
tau2 = mean(z2);
tauSum = tau1 + tau2; %Two Turbines
fprintf('Estimated Expected Combined Power (2 turbines): %.4f W\n', tauSum);
% running_total = cumsum(z1 + z2) ./ (1:N);
% figure;
% plot(1:N, running_total, 'LineWidth', 2);
% xlabel('Number of Samples');
% ylabel('Running Total Expected Power (W)');
% title('Convergence of Combined Expected Power Estimate');
% grid on;


%%  3(b):
k = 1.96;
lambda = 9.13;
alpha = 0.638;
p = 3;
q = 1.5;
fx = @(x) wblpdf(x, lambda, k);
Fx = @(x) wblcdf(x, lambda, k);
fxy = @(x,y) fx(x).*fx(y) .* (1 + alpha * (1 - Fx(x).^p).^(q-1) .* (1 - Fx(y).^p).^(q-1) ...
           .* ((Fx(x).^p).*(1+p*q)-1) .* ((Fx(y).^p).*(1+p*q)-1)); % bivariate pdf
phi = @(x) P(x);

% Find proper g
xgrid = 0:0.5:30;
[Xgrid, Ygrid] = meshgrid(xgrid, xgrid);
% Evaluate φ(x)*φ(y)*fxy(x,y)
temp = phi(Xgrid(:)) .* phi(Ygrid(:)) .* fxy(Xgrid(:), Ygrid(:));
[M, idx] = max(temp);
mu_val = Xgrid(idx);    % use the x-value at the maximum 
mu_biv = [mu_val, mu_val];
sigma_biv = eye(2)*20; % Scaling 
N = 10000;
num = mvnrnd(mu_biv, sigma_biv, N);
X = num(:,1);
Y = num(:,2);
gx_biv = mvnpdf([X Y], mu_biv, sigma_biv);
P1 = P(X);
P2 = P(Y);

xplot = 0:0.5:30;
[Xplot, Yplot] = meshgrid(xplot, xplot);
target = reshape(phi(Xplot(:)), size(Xplot)) .* reshape(phi(Yplot(:)), size(Yplot)) .* fxy(Xplot, Yplot);
% Compute the instrumental density (bivariate normal) on the same grid
g_biv_plot = mvnpdf([Xplot(:) Yplot(:)], mu_biv, sigma_biv);
g_biv_plot = reshape(g_biv_plot, size(Xplot));

figure;
contourf(Xplot, Yplot, target, 20, 'LineColor', 'none');
hold on;
contour(Xplot, Yplot, g_biv_plot, 20, 'LineColor', 'r', 'LineWidth', 1.5);
colorbar;
xlabel('Wind Speed (m/s)');
ylabel('Wind Speed (m/s)');
title('\phi(x)\phi(y)f_{XY}(x,y) and Instrumental Density g');
legend('Target function: \phi(x)\phi(y)f_{XY}(x,y)', 'Instrumental density g', 'Location', 'best');
hold off;

% 'estimate 
P12Exp = mean((P1 .* P2) .* (fxy(X, Y) ./ gx_biv));
cov_est = P12Exp - (7532859.6129/2) *(7532859.6129/2) ;
fprintf('Estimated Covariance of Produced Power: %.4f W^2\n', cov_est);


%% 3(c) 


N = 10000;
v = wblrnd(lambda, k, [N, 1]);  
power_single = P(v);          
var_single = var(power_single);
cov_est = 6628241264731.1680 ;  % results from 3(b) in W^2
var_total = 2*var_single + 2*cov_est;
std_total = sqrt(var_total);
fprintf('Total variance (2 turbines): %.4e W^2\n', var_total);
fprintf('Standard Deviation (2 turbines): %.4e W\n', std_total);

%% 3.d 


x1 = 0:0.5:30;
x2 = 0:0.5:30;
temp = phi(x1) .* phi(x2) .* fxy(x1,x2)';
mu = [12, 12]; % As in 3(b)

% Plot an instrumental density for visualization
sigma0 = eye(2) * 20;
gx = mvnpdf([x1', x1'], mu, sigma0);
figure;
plot(x1, temp, 'b-', 'LineWidth', 2);
hold on;
% Scale gx 
plot(x1, gx * 1e14, 'r--', 'LineWidth', 2);
xlabel('Wind Speed (m/s)');
ylabel('Value');
title('Comparison of Target Function and Instrumental Density g');
legend('Target function', 'g(x) (scaled)', 'Location', 'best');
grid on;
hold off; 
sigma1 = eye(2) * 20;   %  < 9.5e6)
sigma2 = eye(2) * 20   % > 9.5e6)
N = 10000;
sqN = sqrt(N);
se = 1.96;  

% Generate samples
X1 = mvnrnd(mu, sigma1, N);
g1 = mvnpdf(X1, mu, sigma1);
X2 = mvnrnd(mu, sigma2, N);
g2 = mvnpdf(X2, mu, sigma2);
P1 = P(X1(:,1));
P2 = P(X1(:,2));
P1_2 = P(X2(:,1));
P2_2 = P(X2(:,2));
tau_b = mean( ((P1 + P2) < 9.5e6) .* ( fxy(X1(:,1), X1(:,2)) ./ g1 ) );
tau_a = mean( ((P1_2 + P2_2) > 9.5e6) .* ( fxy(X2(:,1), X2(:,2)) ./ g2 ) );
% Compute 95% CI
above_U = tau_a + abs(se * std((P1_2 + P2_2) > 9.5e6) / sqN);
above_L = tau_a - abs(se * std((P1_2 + P2_2) > 9.5e6) / sqN);
above_W = above_U - above_L;
below_U = tau_b + abs(se * std((P1 + P2) < 9.5e6) / sqN);
below_L = tau_b - abs(se * std((P1 + P2) < 9.5e6) / sqN);
below_W = below_U - below_L;
fprintf('Probability (P1+P2 > 9.5e6): %.4f, 95%% CI: [%.4f, %.4f]\n', tau_a, above_L, above_U);
fprintf('Probability (P1+P2 < 9.5e6): %.4f, 95%% CI: [%.4f, %.4f]\n', tau_b, below_L, below_U);
fprintf('Sum of probabilities: %.4f\n', tau_a + tau_b);