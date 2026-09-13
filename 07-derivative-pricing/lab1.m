%% 3.1 European call in a 3-period binomial model and tree model
clear; clc;

S0    = 90;
K     = 100;
r     = 0.05;
sigma = 0.2;
T     = 1;
n     = 3;         % Time steps 

dt = T/n;
u  = exp(sigma*sqrt(dt));
d  = 1/u;
q  = (exp(r*dt) - d) / (u - d);

% Method 1: RNVF 
w  = binopdf(0:n, n, q);                    % probabilities
ST = S0 * (u.^(0:n)) .* (d.^(n:-1:0));      % all possible payoff prices
pay = max(ST - K, 0);
call_RNVF = exp(-r*T) * sum(w .* pay);

% Method 2: Value tree
S = zeros(n+1,n+1);
S(1,1) = S0;
for j = 2:n+1
    for i = 1:j
        S(i,j) = S0 * u^(j-i) * d^(i-1);     % Same with tree=@(n,T,S0,sigma) triu(exp(sigma*sqrt(T/n)*(2*(cumsum(ones(n+1))-1)-(0:n)))*S0)
    end
end

% Pay off at maturity
V = zeros(n+1,n+1);
for i = 1:n+1
    V(i,n+1) = max(S(i,n+1) - K, 0);
end
df = exp(-r*dt); % Discount factor
% Discounted risk neutral back to t=0
for j = n:-1:1
    for i = 1:j
        V(i,j) = df * ( q*V(i,j+1) + (1-q)*V(i+1,j+1) );
    end
end
call_Tree = V(1,1);

fprintf('Inputs: S0=%.2f, K=%.2f, r=%.3f, sigma=%.3f, T=%g, n=%d\n', ...
        S0,K,r,sigma,T,n);
fprintf('u=%.6f, d=%.6f, q=%.6f\n', u,d,q);
fprintf('European Call (RNVF) : %.6f\n', call_RNVF);
fprintf('European Call (Tree) : %.6f\n', call_Tree);
fprintf('Abs. difference      : %.3e\n', abs(call_RNVF - call_Tree));


%% 3.2 American call and put
clear; clc;

S0    = 90;
K     = 100;
r     = 0.05;          
sigma = 0.2;
T     = 1;
Nlist = [1, 3, 10];    % periods to test

% Payer's forward value at t=0
PiF = S0 - K*exp(-r*T);

fprintf('Params: S0=%.2f, K=%.2f, r=%.3f, sigma=%.3f, T=%g\n',S0,K,r,sigma,T);
fprintf('Payer''s forward value Pi_F = %.6f\n\n',PiF);

fprintf('%6s  %12s  %12s  %12s  %12s  %9s  %9s\n', ...
        'N','Call(A)','Put(A)','Call-Put','S0-K','<=?','<= Pi_F?');
for N = Nlist
    [CA, PA] = american_binom(S0, K, r, sigma, T, N);  % American call & put

    diffCAPA = CA - PA;            % C_A - P_A
    leftBnd  = S0 - K;             % S0 - K
    rightBnd = PiF;                % S0 - K*exp(-rT)

    okLeft  = diffCAPA >= leftBnd - 1e-12;   % small tolerance
    okRight = diffCAPA <= rightBnd + 1e-12;

    fprintf('%6d  %12.6f  %12.6f  %12.6f  %12.6f  %9s  %9s\n', ...
            N, CA, PA, diffCAPA, leftBnd, tfstr(okLeft), tfstr(okRight));
end

% functions
function s = tfstr(tf)
    s = "OK";
    if ~tf, s = "FAIL"; end
end

function [CallA, PutA] = american_binom(S0,K,r,sigma,T,N)
    
    dt = T/N;
    u  = exp(sigma*sqrt(dt));
    d  = 1/u;
    q  = (exp(r*dt) - d)/(u - d);
    disc = exp(-r*dt);

    % Stock tree 
    S = zeros(N+1,N+1);
    S(1,1) = S0;
    for j = 2:N+1
        for i = 1:j
            S(i,j) = S0 * u^(j-i) * d^(i-1);
        end
    end

    % American CALL
    V = zeros(N+1,N+1);
    % terminal payoff
    for i = 1:N+1
        V(i,N+1) = max(S(i,N+1) - K, 0);
    end
    % roll back with early-exercise check
    for j = N:-1:1
        for i = 1:j
            cont = disc*( q*V(i,j+1) + (1-q)*V(i+1,j+1) );
            exer = max(S(i,j) - K, 0);
            V(i,j) = max(cont, exer);
        end
    end
    CallA = V(1,1);

    % American PUT 
    V = zeros(N+1,N+1);
    for i = 1:N+1
        V(i,N+1) = max(K - S(i,N+1), 0);
    end
    for j = N:-1:1
        for i = 1:j
            cont = disc*( q*V(i,j+1) + (1-q)*V(i+1,j+1) );
            exer = max(K - S(i,j), 0);
            V(i,j) = max(cont, exer);
        end
    end
    PutA = V(1,1);
end 

%% 3.3 Convergence of European Call
clear; clc;

S0    = 90;
K     = 100;
r     = 0.05;                  % Parameters
sigma = 0.2;
T     = 1;

% Number of binomial steps from 1 to 100
Nvec = 1:100;
priceBin = zeros(size(Nvec));

% RNVF price 
for idx = 1:numel(Nvec)
    N  = Nvec(idx);
    dt = T/N;
    u  = exp(sigma*sqrt(dt));
    d  = 1/u;
    q  = (exp(r*dt) - d)/(u - d);
    disc = exp(-r*T);
    w = binopdf(0:N, N, q); 
    

    % Terminal stock prices for k=0..N (k up-moves)
    % ST(k) = S0 * u^k * d^(N-k)
    ST  = S0 * (u.^(0:N)) .* (d.^(N:-1:0));

    pay = max(ST - K, 0);
    priceBin(idx) = disc * sum(w .* pay);
end

% BS price 
priceBS = bs_call(S0,K,r,sigma,T);

% Print a few values and error 
disp('   N      Binomial     BlackScholes     AbsError');
for idx = [1 2 3 5 10 25 50 100]
    fprintf('%4d   %10.6f     %10.6f     %10.6f\n', ...
        idx, priceBin(Nvec==idx), priceBS, abs(priceBin(Nvec==idx)-priceBS));
end

% Plots convergence and error
figure('Color','w');
subplot(1,2,1);
plot(Nvec, priceBin, 'o-', 'LineWidth',1.2); hold on;
yline(priceBS,'r--','LineWidth',1.2);
xlabel('Number of periods N'); ylabel('Call price');
title('Convergence of Binomial Price to Black–Scholes');
legend('Binomial (RNVF)','Black–Scholes','Location','best'); grid on;

subplot(1,2,2);
semilogy(Nvec, abs(priceBin - priceBS), 'o-','LineWidth',1.2);
xlabel('Number of periods N'); ylabel('|Binomial - BS|');
title('Absolute Error (log scale)'); grid on;

% BS function 
function C = bs_call(S0,K,r,sigma,T)
    d1 = (log(S0/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T));
    d2 = d1 - sigma*sqrt(T);
    C  = S0*normcdf(d1) - K*exp(-r*T)*normcdf(d2);
end

%% 3.4 Convergence of American put (binomial tree)
clear; clc;


S0    = 90;
K     = 100;
r     = 0.05;         
sigma = 0.2;
T     = 1;
Nvec = 1:100;                 % number of periods to test
PA   = zeros(size(Nvec));     

% Black–Scholes (European) put price for reference (lower bound)
P_BS = bs_put(S0,K,r,sigma,T);   % should be about 11.4927
fprintf('Black–Scholes (European) Put: %.6f\n\n', P_BS);

% Compute American put via binomial tree for each N
for k = 1:numel(Nvec)
    N     = Nvec(k);
    PA(k) = american_put_binom(S0,K,r,sigma,T,N);
end


disp('   N      AmericanPut     (P_A - P_BS (European))');
for k = [1 2 3 5 10 25 50 100]
    idx = find(Nvec==k,1);
    fprintf('%4d     %11.6f     %+11.6f\n', k, PA(idx), PA(idx)-P_BS);
end

figure('Color','w');
subplot(1,2,1);
plot(Nvec, PA, 'o-', 'LineWidth',1.2); hold on;
yline(P_BS,'r--','LineWidth',1.2);
xlabel('Number of periods N'); ylabel('American put price');
title('Convergence of American Put (Binomial Tree)');
legend('American Put (binomial)','European Put (BS)','Location','best'); grid on;

subplot(1,2,2);
semilogy(Nvec, max(PA - P_BS, 0), 'o-','LineWidth',1.2);
xlabel('Number of periods N'); ylabel('American premium = P_A - P_{BS}');
title('Early-exercise premium vs N (log scale)'); grid on;

% functions 

function P = bs_put(S0,K,r,sigma,T)
    % Black–Scholes European put 
    d1 = (log(S0/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T));
    d2 = d1 - sigma*sqrt(T);
    P  = K*exp(-r*T)*normcdf(-d2) - S0*normcdf(-d1);
end

function PutA = american_put_binom(S0,K,r,sigma,T,N)
    % American put via recombining binomial tree with early-exercise check
    dt   = T/N;
    u    = exp(sigma*sqrt(dt));
    d    = 1/u;
    q    = (exp(r*dt) - d)/(u - d);
    disc = exp(-r*dt);

    % Stock tree
    S = zeros(N+1,N+1);
    S(1,1) = S0;
    for j = 2:N+1
        for i = 1:j
            S(i,j) = S0 * u^(j-i) * d^(i-1);
        end
    end

    % Terminal payoffs (put)
    V = zeros(N+1,N+1);
    for i = 1:N+1
        V(i,N+1) = max(K - S(i,N+1), 0);
    end

    % Backward induction with early-exercise
    for j = N:-1:1
        for i = 1:j
            cont = disc*( q*V(i,j+1) + (1-q)*V(i+1,j+1) );
            exer = max(K - S(i,j), 0); 
            V(i,j) = max(cont, exer);
        end
    end

    PutA = V(1,1);
end

%Convergence is slower for American put since you can only exercise at
%discrete times.