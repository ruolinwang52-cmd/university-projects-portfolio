%% Question 3
max_n = 10;    
N = 100000;      
estimated_cn = zeros(max_n, 1);
for n = 1:max_n
    N_SA = 0;  % Set Counter for SAWs
    
    parfor i = 1:N  
        % Generate a random walk of n steps
        pos = zeros(n+1, 2);  
        current = [0, 0];
        pos(1, :) = current;
        is_SAW = true;
        
        for k = 1:n
            % Random direction (1=up, 2=down, 3=left, 4=right)
            dir = randi(4);
            switch dir
                case 1  
                    step = [0, 1];
                case 2
                    step = [0, -1];
                case 3
                    step = [-1, 0];
                case 4
                    step = [1, 0];
            end
            current = current + step;
            pos(k+1, :) = current;
        end
        
     
        if size(unique(pos, 'rows'), 1) == n + 1   % Check if all positions are unique
            N_SA = N_SA + 1;
        end
    end
    
    % Estimate c_n(2) as (N_SA / N) * 4^n
    estimated_cn(n) = (N_SA / N) * 4^n;
    fprintf('n = %d, Estimated c_n = %.2f\n', n, estimated_cn(n));
end
figure;
plot(1:max_n, estimated_cn, 'o-', 'LineWidth', 2);
xlabel('Walk length (n)');
ylabel('Estimated c_n(2)');
title('SIS Estimate of Self-Avoiding Walk Counts');
grid on;

%% Question 4
max_n = 18;     
N = 100000;      
weights = zeros(N, max_n);  %Setup 

parfor i = 1:N
    current = [0, 0];
    visited = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    visited(sprintf('%d,%d', current)) = true;
    omega = 1;  % ω_0 = 1
    
    for k = 1:max_n
  
        neighbors = current + [0,1; 0,-1; 1,0; -1,0];
        available = [];
        
      
        for j = 1:4   % Check unvisited neighbors
            pos = neighbors(j, :);
            key = sprintf('%d,%d', pos(1), pos(2));
            if ~visited.isKey(key)
                available = [available; pos];
            end
        end
        
        d_k = size(available, 1);
        if d_k == 0
            weights(i, k:end) = 0; % Trapped, set future weights to 0
            break;
        else
       
            omega = omega * d_k;
            weights(i, k) = omega;
            
      
            chosen_idx = randi(d_k);
            new_pos = available(chosen_idx, :);
            visited(sprintf('%d,%d', new_pos)) = true;
            current = new_pos;
        end
    end
end


estimated_cn = mean(weights, 1);
figure;
semilogy(1:max_n, estimated_cn, 'o-', 'LineWidth', 2);
xlabel('Walk Length (n)');
ylabel('Estimated c_n(2)');
title('SIS for SAWs (Validated Implementation)');
grid on;
%% Question 5
max_n = 18;    
N = 100000;       
resample_threshold = 0.5; % Resample threshold 
particles = struct('pos', cell(N,1), 'visited', cell(N,1), 'weight', cell(N,1));
for i = 1:N
    particles(i).pos = [0, 0];
    particles(i).visited = containers.Map;
    particles(i).visited('0,0') = true;
    particles(i).weight = 1;
end
estimated_cn = ones(1, max_n);  % c_0 = 1

for k = 1:max_n
    % Mutation Step 
    parfor i = 1:N
        if particles(i).weight == 0
            continue;
        end
        
        current = particles(i).pos;
        visited = particles(i).visited;
        
        neighbors = [current + [0,1]; current + [0,-1];
                     current + [1,0]; current + [-1,0]];
        available = [];
        
        for j = 1:4
            pos = neighbors(j,:);
            key = sprintf('%d,%d', pos(1), pos(2));
            if ~visited.isKey(key)
                available = [available; pos];
            end
        end
        
        d_k = size(available, 1);
        
        if d_k == 0
            particles(i).weight = 0;
        else
            chosen_idx = randi(d_k);
            new_pos = available(chosen_idx, :);
            key_new = sprintf('%d,%d', new_pos(1), new_pos(2));
            visited(key_new) = true;
            particles(i).pos = new_pos;
            particles(i).weight = particles(i).weight * d_k;
        end
    end
    
    % average weight 
    weights = [particles.weight];
    avg_weight = mean(weights);
    estimated_cn(k) = avg_weight;  % Directly use current step's average
    
    %  Resampling 
    normalized_weights = weights / sum(weights);
    ESS = 1 / sum(normalized_weights.^2);
    
    if ESS < resample_threshold * N
      
        cum_weights = cumsum(normalized_weights);  % Systematic resampling
        u = (0:N-1)/N + rand(1)/N;
        idx = arrayfun(@(x) find(cum_weights >= x, 1), u);
        
        particles = particles(idx);
        [particles.weight] = deal(1); % Reset weights
    end
    
    fprintf('n = %d, Estimated c_n = %.2f\n', k, estimated_cn(k));
end
figure;
semilogy(1:max_n, estimated_cn, 'o-', 'LineWidth', 2);
xlabel('Walk Length (n)');
ylabel('Estimated c_n(2) (log scale)');
title('Corrected SISR Estimate (Stepwise Averages)');
grid on;


%% Question 6: 
%Run 3 Iterations to Store c_n(2) Estimates
max_n = 18;     
N = 10000;     
num_iterations = 3;  
all_estimated_cn = zeros(num_iterations, max_n);  

for iter = 1:num_iterations
    % Initialize local storage
    local_weights = zeros(N, max_n);

    parfor i = 1:N
        current = [0, 0];
        visited = containers.Map('KeyType', 'char', 'ValueType', 'logical');
        visited(sprintf('%d,%d', current)) = true;
        omega = 1;  % ω_0 = 1
        temp_weights = zeros(1, max_n);  % Store individual weight sequence
        
        for k = 1:max_n
            neighbors = current + [0,1; 0,-1; 1,0; -1,0];
            available = [];

            % Check unvisited neighbors
            for j = 1:4
                pos = neighbors(j, :);
                key = sprintf('%d,%d', pos(1), pos(2));
                if ~visited.isKey(key)
                    available = [available; pos];
                end
            end
            
            d_k = size(available, 1);
            if d_k == 0
                temp_weights(k:end) = 0; % Trapped, set future weights to 0
                break;
            else
                omega = omega * d_k;
                temp_weights(k) = omega;

                % Move to new position
                chosen_idx = randi(d_k);
                new_pos = available(chosen_idx, :);
                visited(sprintf('%d,%d', new_pos)) = true;
                current = new_pos;
            end
        end
        local_weights(i, :) = temp_weights;  % Store individual sequence
    end
    
    all_estimated_cn(iter, :) = mean(local_weights, 1);
    
    fprintf('Iteration %d Completed: Estimated c_n(2) stored.\n', iter);
end

%% Estimate Parameters Using Three Sets of c_n(2) Estimates from Question 4

num_iterations = size(all_estimated_cn, 1);  % Use the stored three iterations from Question 4
estimated_params = zeros(num_iterations, 3); 
n = (1:18)';

for iter = 1:num_iterations
    estimated_cn = all_estimated_cn(iter, :)';  % Use the estimates from Question 4
    y_n = log(estimated_cn); 
    X = [ones(length(n),1), n, log(n)]; 

    % Solve linear regression 
    beta = X \ y_n; 

    % Convert back
    log_A2 = beta(1);
    A2 = exp(log_A2); 
    mu2 = exp(beta(2)); 
    gamma2 = beta(3) + 1; 
    estimated_params(iter, :) = [A2, mu2, gamma2];
    
    fprintf('Iteration %d:\n', iter);
    fprintf('Estimated A_2: %.6f\n', A2);
    fprintf('Estimated mu_2: %.6f\n', mu2);
    fprintf('Estimated gamma_2: %.6f\n', gamma2);
    fprintf('--------------------------------\n');
end

% Display summary of estimated parameters
fprintf('Final Summary of %d Iterations:\n', num_iterations);
fprintf('Iteration\tA_2\t\tmu_2\t\tgamma_2\n');
for iter = 1:num_iterations
    fprintf('%d\t\t%.6f\t%.6f\t%.6f\n', iter, estimated_params(iter,1), estimated_params(iter,2), estimated_params(iter,3));
end


%% Question 9
%Code is practically borrowed from Q.6, with some changes to
%estimate c_n(d) with d=5 instead of d=2
%This time we use only one iteration

max_n = 18;     
N = 10000;     
all_estimated_cn = zeros(1, max_n);  

% Initialize local storage
local_weights = zeros(1, max_n);

parfor i = 1:N
    current = zeros(1, 5);  % (5D) position
    visited = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    visited(sprintf('%f,%f,%f,%f,%f', current)) = true;
    omega = 1;  % ω_0 = 1
    temp_weights = zeros(1, max_n);  % Store individual weight sequence
    
    for k = 1:max_n
        % define neighbors (5D)
        neighbors = [];
        for dim = 1:5
            temp_neighbors = current;
            temp_neighbors(dim) = temp_neighbors(dim) + 1;
            neighbors = [neighbors; temp_neighbors];  
            temp_neighbors(dim) = temp_neighbors(dim) - 2;  
            neighbors = [neighbors; temp_neighbors];  
        end
        
        available = [];

        % Check unvisited neighbors
        for j = 1:size(neighbors, 1)
            pos = neighbors(j, :);
            key = sprintf('%f,%f,%f,%f,%f', pos);
            if ~visited.isKey(key)
                available = [available; pos];
            end
        end
        
        d_k = size(available, 1);
        if d_k == 0
            temp_weights(k:end) = 0; % Trapped, set future weights to 0
            break;
        else
            omega = omega * d_k;
            temp_weights(k) = omega;

            % Move to new position
            chosen_idx = randi(d_k);
            new_pos = available(chosen_idx, :);
            visited(sprintf('%f,%f,%f,%f,%f', new_pos)) = true;
            current = new_pos;
        end
    end
    local_weights(i, :) = temp_weights;  % Store individual sequence
end

all_estimated_cn(1, :) = mean(local_weights, 1);

% estimate parameters Using c_n(5)
estimated_params = zeros(1, 3); 
n = (1:18)';

% use result from the single iteration
estimated_cn = all_estimated_cn(1, :)';
y_n = log(estimated_cn); 
X = [ones(length(n),1), n, log(n)]; 

% Solve linear regression 
beta = X \ y_n; 

% Convert back
log_A5 = beta(1);
A5 = exp(log_A5); 
mu5 = exp(beta(2)); 
gamma5 = beta(3) + 1; 
estimated_params(1, :) = [A5, mu5, gamma5];

% estimated parameters
fprintf('Estimated A_2: %.6f\n', A5);
fprintf('Estimated mu_2: %.6f\n', mu5);
fprintf('Estimated gamma_2: %.6f\n', gamma5);
fprintf('--------------------------------\n');

%% same but now with d=10

max_n = 18;     
N = 10000;     
all_estimated_cn = zeros(1, max_n);  

% Initialize local storage
local_weights = zeros(N, max_n);

parfor i = 1:N
    current = zeros(1, 10);  % (10D) position
    visited = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    visited(sprintf('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f', current)) = true;
    omega = 1;  % ω_0 = 1
    temp_weights = zeros(1, max_n);  % Store individual weight sequence
    
    for k = 1:max_n
        % define neighbors (10D)
        neighbors = [];
        for dim = 1:10
            temp_neighbors = current;
            temp_neighbors(dim) = temp_neighbors(dim) + 1;
            neighbors = [neighbors; temp_neighbors];  
            temp_neighbors(dim) = temp_neighbors(dim) - 2;  
            neighbors = [neighbors; temp_neighbors];  
        end
        
        available = [];

        % Check unvisited neighbors
        for j = 1:size(neighbors, 1)
            pos = neighbors(j, :);
            key = sprintf('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f', pos);
            if ~visited.isKey(key)
                available = [available; pos];
            end
        end
        
        d_k = size(available, 1);
        if d_k == 0
            temp_weights(k:end) = 0; % Trapped, set future weights to 0
            break;
        else
            omega = omega * d_k;
            temp_weights(k) = omega;

            % Move to new position
            chosen_idx = randi(d_k);
            new_pos = available(chosen_idx, :);
            visited(sprintf('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f', new_pos)) = true;
            current = new_pos;
        end
    end
    local_weights(i, :) = temp_weights;  % Store individual sequence
end

all_estimated_cn(1, :) = mean(local_weights, 1);

% Estimate parameters Using c_n(10)
estimated_params = zeros(1, 3); 
n = (1:18)';

% Use result from the single iteration
estimated_cn = all_estimated_cn(1, :)';
y_n = log(estimated_cn); 
X = [ones(length(n),1), n, log(n)]; 

% Solve linear regression 
beta = X \ y_n; 

% Convert back
log_A10 = beta(1);
A10 = exp(log_A10); 
mu10 = exp(beta(2)); 
gamma10 = beta(3) + 1; 
estimated_params(1, :) = [A10, mu10, gamma10];

% Estimated parameters
fprintf('Estimated A_10: %.6f\n', A10);
fprintf('Estimated mu_10: %.6f\n', mu10);
fprintf('Estimated gamma_10: %.6f\n', gamma10);




%% Task 10

load('population_2024.mat');
%parameters
A = 0.8; B = 3.8; C = 0.6; D = 0.99; G = 0.8; H = 1.25;
n = 100; % generations
N_vals = [500, 1000, 10000]; % particle sizes

% (uniform) observation density
p = @(x, y) (y >= G * x & y <= H * x) / (H * x - G * x);

% Loop over different particle sizes (N)
for N_idx = 1:length(N_vals)
    N = N_vals(N_idx);
    
    %initialize vectors for filter estimates, intervals and weights
    tau = zeros(1, n + 1);
    taulower = zeros(1, n + 1);
    tauupper = zeros(1, n + 1); 
    w = zeros(N, 1);
    part = C + (D - C) * rand(N, 1); % particles
    
    
    w = arrayfun(@(x) p(x, Y(1)), part); % weights (for first observation)
    tau(1) = sum(part .* w) / sum(w); % estimate (for first generation)
    
    % confidence interval for generation 0
    [xx, I] = sort(part); % sort data (particles)
    cw = cumsum(w(I)) / sum(w); % cumulative normalized weightsum
    Ilower = find(cw >= 0.025, 1); % index for lower 2.5% quantile
    Iupper = find(cw >= 0.975, 1); % index for upper 2.5% quantile
    taulower(1) = xx(Ilower); % lower 2.5% quantile
    tauupper(1) = xx(Iupper); % upper 2.5% quantile
    
    % Particle filter main loop
    for k = 1:n
        Rk = A + (B - A) * rand; % Sample from U(A, B)
        part = Rk * part .* (1 - part) + randn(N, 1) * sqrt(1 / (1 - A^2)); % initialization
        
        w = arrayfun(@(x) p(x, Y(k + 1)), part); % weights for the current observation
       
        tau(k + 1) = sum(part .* w) / sum(w);  % estimate filtered expectation 
        
        %confidence intervals
        [xx, I] = sort(part); % Sort particles
        cw = cumsum(w(I)) / sum(w); % cumulative normalized weightsum
                                    % for sorted data
        Ilower = find(cw >= 0.025, 1); % index for lower 2.5% quantile
        Iupper = find(cw >= 0.975, 1); % index for upper 2.5% quantile
        taulower(k + 1) = xx(Ilower); % lower 2.5% quantile
        tauupper(k + 1) = xx(Iupper); % upper 2.5% quantile
        
        %resampling indices and particles
        ind = randsample(N, N, true, w); 
        part = part(ind); 
    end
    
    % save results for each N in a different tau variable
    if N == 500
        tau_500 = tau;
        taulower_500 = taulower;
        tauupper_500 = tauupper;
    elseif N == 1000
        tau_1000 = tau;
        taulower_1000 = taulower;
        tauupper_1000 = tauupper;
    elseif N == 10000
        tau_10000 = tau;
        taulower_10000 = taulower;
        tauupper_10000 = tauupper;
    end
    
    % Display results (estimate, confidence intervals, and if true value is inside the interval)
    disp(['Results for N = ', num2str(N), ' particles:']);
    total_diff = 0; % variable accumulated total absolute difference
    for k = 1:n + 1
        diff = abs(tau(k) - X(k));  % Absolute difference
        % Check if true value is inside confidence interval
        inside_interval = (X(k) >= taulower(k) && X(k) <= tauupper(k)); % Check if true value is inside the CI
        
        % display current generation results
        fprintf('Generation %d: Estimate = %.4f, True Value = %.4f, CI = [%.4f, %.4f], Inside CI = %d\n', ...
            k, tau(k), X(k), taulower(k), tauupper(k), inside_interval);
        
        % get total difference
        total_diff = total_diff + diff;
    end
    
    % display total difference
    fprintf('Total absolute difference for N = %d: %.4f\n\n', N, total_diff);
end
%%
% plotting (one plot for each N) of estimates vs real values
figure()
hold on
plot(1:n+1, tau_500, '--o','LineWidth', 1);
plot(1:n+1, X, '--x', 'LineWidth', 1);
hold off
xlabel('generation')
ylabel('relative population size')
legend('Estimation','X')
title('Filter estimate')

figure()
hold on
plot(1:n+1, tau_1000, '--o','LineWidth', 1);
plot(1:n+1, X, '--x', 'LineWidth', 1);
hold off
xlabel('generation')
ylabel('relative population size')
legend('Estimation','X')
title('Filter estimate')

figure()
hold on
plot(1:n+1, tau_10000, '--o','LineWidth', 1);
plot(1:n+1, X, '--x', 'LineWidth', 1);
hold off
xlabel('generation')
ylabel('relative population size')
legend('Estimation','X')
title('Filter estimate')
