%% Problem set 1 - QM2 - Ferrari Jimenez
clear;clc;

addpath("0_functions/");

% set up parameters

par.w = 1;
par.sigma = 2;
par.beta = 0.98;
par.b = 0;
par.rho = 0.95;
par.R = (1/par.beta) * 0.995;


par.n = 1;

cpar.N = 5;
cpar.mu = 0;
cpar.sigmae = 0.05;

%% Question 1 

[gri.z, gri.prob] = rouwenhorst(cpar.N, cpar.mu, par.rho, sqrt(cpar.sigmae));

gri.z = exp(gri.z);
z_raw = gri.z;

[V, D] = eig(gri.prob');

[~, idx] = min(abs(diag(D) - 1));
pi_dist = V(:, idx) / sum(V(:, idx));

mean_z = pi_dist' * z_raw;
gri.z = z_raw / mean_z;

fprintf('Transition Matrix P:\n');
disp(gri.prob);
fprintf('Normalized Support z:\n');
disp(gri.z);
fprintf('Verified Mean (should be 1.0): %f\n', pi_dist' * gri.z);

%% Set up parameters for VFI

cpar.tol = 1e-6;
cpar.Nkap = 20;
cpar.min = 0;
cpar.max = 10;

gri.k = linspace(log(cpar.min +1), log(cpar.max +1), cpar.Nkap);

gri.k = exp(gri.k) - 1;

%% Part A - VFI


[V, cpol, kpol] = VFI_GS(par, gri, cpar);

%% Plot

% Create the figure
figure;
hold on;

% 1. Plot the policy function for all 5 productivity states
% Using a loop to assign different colors and dynamic legend entries
colors = lines(5); 
for j = 1:5
    % Assuming rows are k and columns are z. If yours is transposed, use kpol(j, :)
    plot(gri.k, kpol(j, :), 'LineWidth', 2, 'Color', colors(j, :), ...
         'DisplayName', sprintf('Productivity State z_%d', j));
end

% 2. Plot the 45-degree line
% This shows where k' = k (agents maintain their current wealth)
plot(gri.k, gri.k, 'k--', 'LineWidth', 1.5, 'DisplayName', '45-degree line');

hold off;

% 3. Format the plot for clarity
title('Optimal Capital Policy Function: $k''(k,z)$', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Current Asset Holdings ($k$)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Next Period Asset Holdings ($k''$)', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'northwest', 'FontSize', 10);
grid on;

%% Plot the cpol

% Create the figure for Consumption Policy
figure;
hold on;

% 1. Plot the consumption policy function for all 5 productivity states
% Higher productivity states (z) should generally result in higher consumption
colors = lines(5); 
for j = 1:5
    % Plotting current consumption against the asset grid
    % Adjust index to cpol(:, j) if your matrix is (Capital x Productivity)
    plot(gri.k, cpol(j, :), 'LineWidth', 2, 'Color', colors(j, :), ...
         'DisplayName', sprintf('Productivity State $z_%d$', j));
end

% 2. Formatting the plot
% Note: There is no standard "45-degree line" for consumption, 
% as consumption is typically less than total wealth (x).
title('Optimal Consumption Policy Function: $c(k,z)$', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Current Asset Holdings ($k$)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Current Consumption ($c$)', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'northwest', 'FontSize', 10, 'Interpreter', 'latex');
grid on;
hold off;

%% Plotting Value Functions from Question 1.A
figure('Name', 'Value Functions');
hold on;

% Define colors for the 5 productivity states
colors = lines(5); 

% Loop through each productivity state to plot V(k, z)
for j = 1:cpar.N
    % Assuming V is (Nz x Nkap). If yours is (Nkap x Nz), use V(:, j)
    plot(gri.k, V(j, :), 'LineWidth', 2, 'Color', colors(j, :), ...
         'DisplayName', sprintf('Productivity $z_{%d} = %.2f$', j, gri.z(j)));
end

% Formatting
grid on;
set(gca, 'FontSize', 11);
xlabel('Current Assets ($k$)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Value Function $V(k, z)$', 'Interpreter', 'latex', 'FontSize', 12);
title('Optimal Value Functions by Productivity State', 'Interpreter', 'latex', 'FontSize', 14);
legend('Location', 'southeast', 'Interpreter', 'latex');

% Optional: Set x-axis limit to show the curvature at the bottom more clearly
xlim([gri.k(1), gri.k(end)]); 

hold off;


%% Part 3 - Simulating 2000 individuals over time (Fast Vectorized Discrete Version)
numInd = 2000;
time = 1400;

% Matrices
z_idx = zeros(numInd, time);
z_i   = zeros(numInd, time);
c_i   = zeros(numInd, time);
k_i   = zeros(numInd, time);

% Initial Conditions
z_idx(:,1) = 3; 
z_i(:,1)   = gri.z(3);
k_i(:,1)   = 5;

P_cum = cumsum(gri.prob, 2);

% 1. PRE-SIMULATE THE SHOCKS OUTSIDE THE LOOP (Massive Speedup)
% We draw all the random numbers at once and calculate the paths
rng(123); % Set seed for reproducibility
draws = rand(numInd, time);

for t = 1:time-1
    for z_state = 1:cpar.N
        % Find everyone currently in z_state
        mask = (z_idx(:, t) == z_state);
        if any(mask)
            % Fast transition: count how many cumulative probabilities the random draw exceeds
            next_state = sum(draws(mask, t) > P_cum(z_state, :), 2) + 1;
            z_idx(mask, t+1) = next_state;
        end
    end
end

% Map the indices to actual numerical values for all periods at once
for z_state = 1:cpar.N
    z_i(z_idx == z_state) = gri.z(z_state);
end

% 2. VECTORIZED SIMULATION LOOP
%
% NOTE ON POLICY EVALUATION OUTSIDE THE GRID:
% The policy k'(k,z) was solved only on gri.k = [0, 10]. Outside this 
% region the true policy is concave and flattens (savings rate decreases 
% as wealth grows), but linear extrapolation incorrectly projects a steep 
% slope, causing k' to overshoot and compound each period. We therefore 
% clamp all queries to [gri.k(1), gri.k(end)]. At the upper bound this 
% is conservative: it slightly understates k' for the richest agents, 
% which is preferable to an explosive trajectory with no theoretical basis.

k_lo = gri.k(1);
k_hi = gri.k(end);

for t = 1:time
    
    % Instead of looping over 2000 individuals, we loop over the 5 states
    for z_state = 1:cpar.N
        
        % Find all individuals in this specific productivity state today
        mask = (z_idx(:, t) == z_state);
        
        if any(mask)
            
            % Policy function row for this productivity state (size: [5 x 20])
            current_kpol = kpol(z_state, :);
            
            % Clamp capital to the grid to avoid unbounded extrapolation
            k_query = min(max(k_i(mask, t), k_lo), k_hi);
            
            % Interpolate k'(k,z) on the grid (no extrapolation)
            k_next = interp1(gri.k, current_kpol, k_query, 'linear');
            
            % Enforce the borrowing constraint: k' >= 0
            k_next = max(0, k_next);
            
            % Consumption from the budget constraint: c = R*k + z*w - k'
            c_i(mask, t) = par.R * k_i(mask, t) + gri.z(z_state) * par.w - k_next;
            c_i(mask, t) = max(1e-10, c_i(mask, t)); % numerical safety
            
            if t < time
                k_i(mask, t+1) = k_next;
            end
        end
    end
end

% Extract the stationary distribution (periods 401 to 1400)
k_steady = k_i(:, 401:1400);

% Flatten the matrix into a single vector containing all observations
k_vec = k_steady(:); 

% Calculate statistics across all individuals and post-burn-in periods
fprintf('Mean k across individuals (periods 401-1400): %.4f\n', mean(k_vec));
fprintf('Max  k across individuals (periods 401-1400): %.4f\n', max(k_vec));
%% Plotting the Wealth Distribution and Calculating Statistics

% 1. Discard the first 400 periods (burn-in) 
drop = 400;
k_steady = k_i(:, drop+1:end);

% Flatten the matrix into a single vector of stationary asset holdings (pooling periods 401-1400)
k_vec = k_steady(:);

% 2. Plot a histogram of the wealth distribution
figure;
% Using 100 bins for a smooth look, normalized to form a probability density
histogram(k_vec, 100, 'Normalization', 'pdf', 'FaceColor', [0.2 0.4 0.6], 'EdgeColor', 'none');
hold on;

% Optional: Overlay a Kernel Density Estimate (uncomment the next two lines to use)
% [f_kde, xi_kde] = ksdensity(k_vec);
% plot(xi_kde, f_kde, 'r-', 'LineWidth', 2, 'DisplayName', 'Kernel Density');

title('Stationary Wealth Distribution', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Density', 'Interpreter', 'latex', 'FontSize', 12);
grid on;

% 3. Calculate Required Statistics (Gini, 90/10, and 99/1 ratios)
% Sort wealth for the Gini calculation
k_sorted = sort(k_vec);
N_k = length(k_sorted);

% Gini Coefficient Formula
gini = (2 * sum((1:N_k)' .* k_sorted)) / (N_k * sum(k_sorted)) - (N_k + 1) / N_k;

% Percentiles
p90 = prctile(k_vec, 90);
p10 = prctile(k_vec, 10);
p99 = prctile(k_vec, 99);
p1  = prctile(k_vec, 1);

% Ratios
ratio_90_10 = p90 / p10;
ratio_99_1  = p99 / p1;

% Display Results in the Command Window
fprintf('--- Wealth Distribution Statistics ---\n');
fprintf('Mean Asset Holdings: %.4f\n', mean(k_vec));
fprintf('Gini Coefficient: %.4f\n', gini);
fprintf('90/10 Percentile Ratio: %.4f\n', ratio_90_10);
fprintf('99/1 Percentile Ratio: %.4f\n', ratio_99_1);
%% Euler Errors

c_implied = zeros(cpar.N,cpar.Nkap);
EE = zeros(cpar.N,cpar.Nkap);

par.imp = 0.995^(-1/par.sigma);

% 4. Calculate Euler Errors for each individual
for j = 1:cpar.N
    for k = 1:cpar.Nkap

        c_now = cpol(j, k);
        k_next = kpol(j, k);

        % 1. Calculate Expected Marginal Utility: E[ c(k', z')^(-sigma) | z ]
        expected_MU = 0;
        for next_j = 1:cpar.N

            c_next = interp1(gri.k, cpol(next_j, :), k_next, 'linear', 'extrap');
            if c_next <= 0
               warning('Consumption is zero or negative at k_next = %f', k_next);
               c_next = 1e-10; % Tiny floor to prevent crash
            end
            prob_transition = gri.prob(j, next_j); 
            
            % The expectation applies to c^(-sigma), NOT just c
            expected_MU = expected_MU + prob_transition * (c_next^(-par.sigma));
        end
        
        % 2. Apply your simplified constant 0.995
        c_implied(j,k) = (0.995 * expected_MU)^(-1/par.sigma);
        
        % 3. Calculate the absolute percentage error
        EE(j,k) = abs((c_implied(j,k) - c_now) / c_now) * 100;
        
    end
end


%% Plotting the Euler Equation Errors

figure;
hold on;
colors = lines(cpar.N); 

% We plot the log10 of the errors for each productivity state
for j = 1:cpar.N
    % Note: We add a tiny number (1e-16) to EE to avoid log10(0) if the error is exactly zero
    plot(gri.k, log10(EE(j, :) + 1e-16), 'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', sprintf('Productivity State z_%d', j));
end
hold off;

% Formatting the Plot
title('Euler Equation Errors', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$\log_{10}$ Absolute Percentage Error', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;

% The problem set specifically asks to "Report the maximum across i and j"
max_EE_err = max(EE(:));
fprintf('--- Euler Equation Error Check ---\n');
fprintf('Maximum Euler Error (Absolute %%): %f%%\n', max_EE_err);

%% 1.B - EGM
% NOTE: gri.x is built internally by EGM.m, no need to construct it here.


solving_EGM = EGM(par, cpar, gri);
gri.x = solving_EGM.x;  % retrieve the x-grid built by EGM


%% Plots  x

figure('Name', 'EGM Policy Function: Cash-on-Hand');
hold on;
colors = lines(cpar.N);

for j = 1:cpar.N
    plot(gri.x, solving_EGM.kpol_x(:, j), 'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', ['z = ', num2str(gri.z(j))]);
end

% Reference line: saving everything (k' = x)
plot(gri.x, gri.x, 'k--', 'HandleVisibility', 'off'); 

xlabel('Current Cash-on-Hand ($x$)', 'Interpreter', 'latex');
ylabel('Next Period Assets ($k''$)', 'Interpreter', 'latex');
title('Optimal Savings Policy $k''(x, z)$ (EGM)', 'Interpreter', 'latex');
legend('Location', 'northwest');
grid on;

%% Plots k

figure('Name', 'EGM Policy Function: Assets');
hold on;

for j = 1:cpar.N
    plot(gri.k, solving_EGM.kpol_k(:, j), 'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', ['z = ', num2str(gri.z(j))]);
end

% 45-degree line: where k' = k
plot(gri.k, gri.k, 'k--', 'LineWidth', 1.5, 'DisplayName', '45-degree line');

xlabel('Current Assets ($k$)', 'Interpreter', 'latex');
ylabel('Next Period Assets ($k''$)', 'Interpreter', 'latex');
title('Optimal Savings Policy $k''(k, z)$ (EGM)', 'Interpreter', 'latex');
legend('Location', 'northwest');
grid on;

%% ===== VFI vs EGM: Policy Function Comparison =====

% The VFI stores policies as (Nz x Nk), EGM stores them as (Nk x Nz).
% Transpose VFI policies so both are (Nk x Nz) for direct comparison.

kpol_VFI = kpol';    % now (Nk x Nz)
cpol_VFI = cpol';    % now (Nk x Nz)

kpol_EGM = solving_EGM.kpol_k;  % already (Nk x Nz)
cpol_EGM = solving_EGM.cpol_k;

% --- Savings Policy Overlay ---
figure('Name', 'VFI vs EGM: Savings Policy');
hold on;
colors = lines(cpar.N);
for j = 1:cpar.N
    plot(gri.k, kpol_VFI(:, j), '-',  'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', sprintf('VFI  z_%d', j));
    plot(gri.k, kpol_EGM(:, j), '--', 'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', sprintf('EGM  z_%d', j));
end
plot(gri.k, gri.k, 'k:', 'LineWidth', 1, 'DisplayName', '45-degree');
hold off;
title('Savings Policy: VFI (solid) vs EGM (dashed)', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Current Assets ($k$)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$k''(k,z)$', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'northwest', 'FontSize', 8);
grid on;

% --- Consumption Policy Overlay ---
figure('Name', 'VFI vs EGM: Consumption Policy');
hold on;
for j = 1:cpar.N
    plot(gri.k, cpol_VFI(:, j), '-',  'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', sprintf('VFI  z_%d', j));
    plot(gri.k, cpol_EGM(:, j), '--', 'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', sprintf('EGM  z_%d', j));
end
hold off;
title('Consumption Policy: VFI (solid) vs EGM (dashed)', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Current Assets ($k$)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$c(k,z)$', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'northwest', 'FontSize', 8);
grid on;

% --- Quantitative Differences ---
diff_k = abs(kpol_VFI - kpol_EGM);
diff_c = abs(cpol_VFI - cpol_EGM);

fprintf('\n--- VFI vs EGM Policy Differences ---\n');
fprintf('Savings policy  sup-norm:  %.6f\n', max(diff_k(:)));
fprintf('Savings policy  L2-norm:   %.6f\n', sqrt(mean(diff_k(:).^2)));
fprintf('Consumption pol sup-norm:  %.6f\n', max(diff_c(:)));
fprintf('Consumption pol L2-norm:   %.6f\n', sqrt(mean(diff_c(:).^2)));

%% ===== EGM Euler Equation Errors =====

c_implied_EGM = zeros(cpar.Nkap, cpar.N);
EE_EGM        = zeros(cpar.Nkap, cpar.N);

for j = 1:cpar.N
    for i = 1:cpar.Nkap

        c_now  = cpol_EGM(i, j);
        k_next = kpol_EGM(i, j);

        % Expected marginal utility: E[ c(k', z')^(-sigma) | z_j ]
        expected_MU = 0;
        for next_j = 1:cpar.N
            % Tomorrow's consumption from EGM policy (column = z-state)
            c_next = interp1(gri.k, cpol_EGM(:, next_j), k_next, 'linear', 'extrap');
            c_next = max(c_next, 1e-10);
            expected_MU = expected_MU + gri.prob(j, next_j) * (c_next^(-par.sigma));
        end

        % Euler-implied consumption
        c_implied_EGM(i, j) = (par.beta * par.R * expected_MU)^(-1/par.sigma);

        % Percentage error
        EE_EGM(i, j) = abs((c_implied_EGM(i, j) - c_now) / c_now) * 100;
    end
end

% Plot
figure('Name', 'Euler Errors: VFI vs EGM');
hold on;
for j = 1:cpar.N
    plot(gri.k, log10(EE(j, :) + 1e-16), '-',  'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', sprintf('VFI  z_%d', j));
    plot(gri.k, log10(EE_EGM(:, j)' + 1e-16), '--', 'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', sprintf('EGM  z_%d', j));
end
hold off;
title('Euler Equation Errors: VFI (solid) vs EGM (dashed)', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$\log_{10}$ Absolute \% Error', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 8);
grid on;

max_EE_EGM = max(EE_EGM(:));
fprintf('\n--- EGM Euler Equation Error Check ---\n');
fprintf('Maximum EGM Euler Error (Absolute %%): %f%%\n', max_EE_EGM);
fprintf('Maximum VFI Euler Error (Absolute %%): %f%%\n', max_EE_err);


%% Question 2 - General Equilibirum
% set initial min and max for R
R_min = 0.8; % Re-allow R to be < 1. It can go very low because MPK = R natively.
R_max = 1/par.beta;
% WE WERE NOT GIVEN THE ALPHA IN THE PROBLEM SET
par.alpha = 0.33;
% set up loop parameters
tol = 1e-3;
diff = tol +1;
iter = 0;
max_iter = 100;
% set up new simulation parameters
time_GE = 300;
% Pre-draw discrete Markov shocks (same method as Q1.A.3)
z_idx_GE = zeros(numInd, time_GE);
z_idx_GE(:, 1) = 3;  % start everyone at median productivity state
P_cum_GE = cumsum(gri.prob, 2);
rng(123);
draws_GE = rand(numInd, time_GE);
for t = 1:time_GE-1
    for z_state = 1:cpar.N
        mask = (z_idx_GE(:, t) == z_state);
        if any(mask)
            z_idx_GE(mask, t+1) = sum(draws_GE(mask, t) > P_cum_GE(z_state, :), 2) + 1;
        end
    end
end
% Start everybody off at the median capital
k_sim = 5 * ones(numInd, 1);
% start off the while loop
fprintf('--- Starting General Equilibrium Bisection ---\n');
while abs(diff) > tol && iter < max_iter 
    iter = iter +1;
    % set the guess for R as the average of min and max guesses
    R_guess = (R_max+R_min)/2;
    % compute demand for capital and wage (formulas found on paper)
    % Reverting to the strict PDF formulation: Marginal Product = R
    
    k_d = (par.alpha / R_guess)^(1/(1-par.alpha));
    w_guess = (1 - par.alpha) * (par.alpha / R_guess)^(par.alpha / (1 - par.alpha));
    par.R = R_guess;
    par.w = w_guess;
    % Rebuild capital grid: k_N = 10*w per PDF instructions
    cpar.max = 10 * par.w;
    gri.k = linspace(log(cpar.min + 1), log(cpar.max + 1), cpar.Nkap);
    gri.k = exp(gri.k) - 1;
    % solve EGM with these new parameters
    solving_EGM = EGM(par, cpar, gri);
    % store the capital policy from EGM on this iteration
    kpol_k = solving_EGM.kpol_k;
    % Simulate using discrete shocks (vectorized by z-state, same as Q1.A.3)
    k_i = zeros(numInd, time_GE);
    k_i(:, 1) = k_sim;
    k_lo = gri.k(1);
    k_hi = gri.k(end);
    for t = 1:time_GE - 1
        for z_state = 1:cpar.N
            mask = (z_idx_GE(:, t) == z_state);
            if any(mask)
                k_query = min(max(k_i(mask, t), k_lo), k_hi);
                k_next = interp1(gri.k, kpol_k(:, z_state), k_query, 'linear');
                k_i(mask, t+1) = max(0, k_next);
            end
        end
    end
    k_sim = k_i(:, end);
    % take the average of individual capital holding to get the capital
    % supply
    k_supply = mean(k_sim);
    % we calculate the difference to see if our guess was too high or too
    % low
    diff = k_supply - k_d;
    fprintf('Iter %2d: R = %.4f | w = %.4f | K_sup = %.3f | K_dem = %.3f | Diff = %.3f\n', ...
             iter, R_guess, w_guess, k_supply, k_d, diff);
    % Update the guesses on the bisection
    if diff > 0 
        R_max = R_guess;
    else
        R_min = R_guess;
    end
    % end the while loop.
end
fprintf('--- General Equilibrium Found! ---\n');
fprintf('Equilibrium Interest Rate R*: %.4f\n', R_guess);
fprintf('Equilibrium Wage w*: %.4f\n', w_guess);
fprintf('Aggregate Capital K*: %.4f\n', k_supply);
%% Plots 
figure('Name', 'GE Wealth Distribution');
hold on;
% Plot normalized histogram
histogram(k_sim, 60, 'Normalization', 'pdf', 'FaceColor', [0.2 0.4 0.6], ...
    'EdgeColor', 'w', 'DisplayName', 'Histogram');
% Overlay Kernel Density Estimate for a smooth curve
[f_kde, xi_kde] = ksdensity(k_sim);
plot(xi_kde, f_kde, 'r-', 'LineWidth', 2, 'DisplayName', 'Kernel Density');
title('Stationary Wealth Distribution ($R^*$)', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Density', 'Interpreter', 'latex', 'FontSize', 12);
legend('Location', 'northeast');
grid on;
hold off;
%% Inequality Statistics
% Sort wealth to calculate Gini
wealth_sorted = sort(k_sim);
N_ind = length(wealth_sorted);
% Gini Coefficient Formula
% G = (2 * sum(i * y_i) / (N * sum(y_i))) - (N + 1) / N
gini_num = sum((1:N_ind)' .* wealth_sorted);
gini_den = N_ind * sum(wealth_sorted);
gini = (2 * gini_num / gini_den) - (N_ind + 1) / N_ind;
% Percentile Ratios
p1  = prctile(k_sim, 1);
p10 = prctile(k_sim, 10);
p90 = prctile(k_sim, 90);
p99 = prctile(k_sim, 99);
ratio_90_10 = p90 / p10;
ratio_99_1  = p99 / p1;
% Display Results
fprintf('\n--- Inequality Statistics (General Equilibrium) ---\n');
fprintf('Gini Coefficient:      %.4f\n', gini);
fprintf('90/10 Percentile Ratio: %.4f\n', ratio_90_10);
fprintf('99/1 Percentile Ratio:  %.4f\n', ratio_99_1);