%% Problem set 1 - QM2 - Ferrari Jimenez
clear; clc; close all;

% =========================================================================
% Formatting Defaults (Added for academic journal style)
% =========================================================================
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontSize', 16);
set(groot, 'defaultTextFontSize', 16);
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesBox', 'on');
set(groot, 'defaultLineLineWidth', 2);

rng(333);

addpath("0_functions/");

% set up parameters
par.w = 1;
par.sigma = 2;
par.beta = 0.98;
par.b = 0;
par.rho = 0.95;
par.R = (1/par.beta) * 0.995;
par.n = 1;
par.delta = 0.025;

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
cpar.min = 1e-6;
cpar.max = 10; %***

gri.k = linspace(log(cpar.min +1), log(cpar.max +1), cpar.Nkap);
gri.k = exp(gri.k) - 1;

%% Part A - VFI
fprintf('Starting VFI...\n');
tic; % Start stopwatch
[V, cpol, kpol] = VFI_GS(par, gri, cpar);
time_VFI = toc; % Stop stopwatch
fprintf('VFI Complete. Time taken: %.2f seconds\n', time_VFI);

%% Plot
% Create the figure
figure;
hold on;
colors = lines(5); 
for j = 1:5
    plot(gri.k, kpol(j, :), 'LineWidth', 2, 'Color', colors(j, :), ...
         'DisplayName', sprintf('Productivity State z_%d', j));
end
plot(gri.k, gri.k, 'k--', 'LineWidth', 1.5, 'DisplayName', '45-degree line');
hold off;

title('Optimal Capital Policy Function: $k''(k,z)$', 'Interpreter', 'latex');
xlabel('Current Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Next Period Asset Holdings ($k''$)', 'Interpreter', 'latex');
legend('Location', 'northwest');
grid on;

%% Plot the cpol
figure;
hold on;
colors = lines(5); 
for j = 1:5
    plot(gri.k, cpol(j, :), 'LineWidth', 2, 'Color', colors(j, :), ...
         'DisplayName', sprintf('Productivity State $z_%d$', j));
end
title('Optimal Consumption Policy Function: $c(k,z)$', 'Interpreter', 'latex');
xlabel('Current Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Current Consumption ($c$)', 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'latex');
grid on;
hold off;

%% Plotting Value Functions from Question 1.A
figure('Name', 'Value Functions');
hold on;
colors = lines(5); 
for j = 1:cpar.N
    plot(gri.k, V(j, :), 'LineWidth', 2, 'Color', colors(j, :), ...
         'DisplayName', sprintf('Productivity $z_{%d} = %.2f$', j, gri.z(j)));
end
grid on;
set(gca);
xlabel('Current Assets ($k$)', 'Interpreter', 'latex');
ylabel('Value Function $V(k, z)$', 'Interpreter', 'latex');
title('Optimal Value Functions by Productivity State', 'Interpreter', 'latex');
legend('Location', 'southeast', 'Interpreter', 'latex');
xlim([gri.k(1), gri.k(end)]); 
hold off;

%% Part 3 - Simulating 2000 individuals over time
numInd = 2000;
time = 1400;

% 1. Pre-simulate the discrete Markov Chain shocks (Indices 1 to 5)
z_idx = zeros(numInd, time);
z_idx(:, 1) = 3; % Start everyone at the median productivity state (index 3)

for t = 1:time-1
    for j = 1:numInd
        probs = gri.prob(z_idx(j, t), :);
        z_idx(j, t+1) = find(rand <= cumsum(probs), 1);
    end
end

% Map the discrete indices back to the actual productivity values
z_i = gri.z(z_idx);

% 2. Matrices for asset dynamics
c_i = zeros(numInd, time);
k_i = zeros(numInd, time);
k_i(:,1) = 5;

% === FASTER 1D INTERPOLANT UPGRADE ===
% Because z is now strictly discrete, we don't need 2D interpolation anymore!
% We just create five separate 1D interpolants (one for each z state).
F_cpol = cell(cpar.N, 1);
F_kpol = cell(cpar.N, 1);
for m = 1:cpar.N
    F_cpol{m} = griddedInterpolant(gri.k, cpol(m, :), 'linear', 'linear');
    F_kpol{m} = griddedInterpolant(gri.k, kpol(m, :), 'linear', 'linear');
end

for t = 1:time
    for j = 1:numInd
        % Identify which discrete state the agent is in today
        current_z_state = z_idx(j, t);
        
        % Query the specific 1D interpolant for that state
        c_i(j,t) = F_cpol{current_z_state}(k_i(j, t));
         
% Calculate next period states
        if t < time
             k_prime = F_kpol{current_z_state}(k_i(j, t));
             
             % REMOVED cpar.max: Enforce ONLY the borrowing constraint (0)
             k_i(j, t+1) = max(0, k_prime); 
        end
    end
end

%% Plotting the Wealth Distribution and Calculating Statistics
drop = 400;
k_steady = k_i(:, drop+1:end);
k_vec = k_steady(:);

figure;
histogram(k_vec, 100, 'Normalization', 'pdf', 'FaceColor', [0.2 0.4 0.6], 'EdgeColor', 'none');
hold on;
title('Stationary Wealth Distribution', 'Interpreter', 'latex');
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Density', 'Interpreter', 'latex');
grid on;

% Calculate Statistics
k_sorted = sort(k_vec);
N_k = length(k_sorted);
gini = (2 * sum((1:N_k)' .* k_sorted)) / (N_k * sum(k_sorted)) - (N_k + 1) / N_k;

p90 = prctile(k_vec, 90);
p10 = prctile(k_vec, 10);
p99 = prctile(k_vec, 99);
p1  = prctile(k_vec, 1);
ratio_90_10 = p90 / p10;
ratio_99_1  = p99 / p1;

fprintf('--- Wealth Distribution Statistics ---\n');
fprintf('Mean Asset Holdings: %.4f\n', mean(k_vec));
fprintf('Gini Coefficient: %.4f\n', gini);
fprintf('90/10 Percentile Ratio: %.4f\n', ratio_90_10);
fprintf('99/1 Percentile Ratio: %.4f\n', ratio_99_1);

%% Euler Errors
c_implied = zeros(cpar.N,cpar.Nkap);
EE = zeros(cpar.N,cpar.Nkap);
par.imp = 0.995^(-1/par.sigma);

% === GRIDDED INTERPOLANT UPGRADE ===
% Pre-build an array of 1D interpolants for expected marginal utility
F_c_euler = cell(cpar.N, 1);
for m = 1:cpar.N
    F_c_euler{m} = griddedInterpolant(gri.k, cpol(m, :), 'linear', 'linear');
end

% 4. Calculate Euler Errors for each individual
for j = 1:cpar.N
    for k = 1:cpar.Nkap
        c_now = cpol(j, k);
        k_next = kpol(j, k);

        % 1. Calculate Expected Marginal Utility
        expected_MU = 0;
        for next_j = 1:cpar.N
            
            % Query the fast 1D interpolant for tomorrow's state
            c_next = F_c_euler{next_j}(k_next);
            
            if c_next <= 0
               c_next = 1e-10; % Tiny floor to prevent crash
            end
            prob_transition = gri.prob(j, next_j); 
            
            expected_MU = expected_MU + prob_transition * (c_next^(-par.sigma));
        end
        
        c_implied(j,k) = (0.995 * expected_MU)^(-1/par.sigma);
        EE(j,k) = abs((c_implied(j,k) - c_now) / c_now) * 100;
        
    end
end

%% Plotting the Euler Equation Errors
figure;
hold on;
colors = lines(cpar.N); 
for j = 1:cpar.N
    plot(gri.k, log10(EE(j, :) + 1e-16), 'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', sprintf('Productivity State z_%d', j));
end
hold off;

title('Euler Equation Errors', 'Interpreter', 'latex');
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('$\log_{10}$ Absolute Percentage Error', 'Interpreter', 'latex');
legend('Location', 'best');
grid on;

max_EE_err = max(EE(:));
fprintf('--- Euler Equation Error Check ---\n');
fprintf('Maximum Euler Error (Absolute %%): %f%%\n', max_EE_err);

%% Grid for x
x_min = min(gri.z)  + par.R * gri.k(1);
x_max = max(gri.z) + par.R * gri.k(end);
gri.x = linspace(x_min, x_max, 100);

%% 1.B - EGM 
fprintf('\nStarting EGM...\n');
tic;
solving_EGM = EGM(par, cpar, gri);
time_EGM = toc;
fprintf('EGM Complete. Time taken: %.2f seconds\n', time_EGM);
fprintf('EGM is %.2f times faster than VFI!\n', time_VFI / time_EGM);

%% Plots x
figure('Name', 'EGM Policy Function: Cash-on-Hand');
hold on;
colors = lines(cpar.N);
for j = 1:cpar.N
    plot(gri.x, solving_EGM.kpol_x(:, j), 'LineWidth', 2, 'Color', colors(j,:), ...
         'DisplayName', ['z = ', num2str(gri.z(j))]);
end
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
plot(gri.k, gri.k, 'k--', 'LineWidth', 1.5, 'DisplayName', '45-degree line');
xlabel('Current Assets ($k$)', 'Interpreter', 'latex');
ylabel('Next Period Assets ($k''$)', 'Interpreter', 'latex');
title('Optimal Savings Policy $k''(k, z)$ (EGM)', 'Interpreter', 'latex');
legend('Location', 'northwest');
grid on;

%% Comparison plots

% Compare the VFI policy and the EGM policy (interpolated back to k-grid)
% Note: kpol from VFI is (z, k). solving_EGM.kpol_k is (k, z). 
% We must transpose one to compare them!
max_pol_diff = max(abs(kpol' - solving_EGM.kpol_k), [], 'all');
fprintf('Maximum difference between VFI and EGM policy functions: %e\n', max_pol_diff);

%% Question 2 - General Equilibirum

% 1. Update the lower bound to strictly prevent negative rental rates
R_min = 1 - par.delta + 1e-4; 
R_max = 1 / par.beta;
par.alpha = 0.33;

tol = 1e-3;
diff = tol + 1;
iter = 0;
max_iter = 100;

time_GE = 300;
z_i = z_i(:, 1:time_GE);
k_sim = 5 * ones(numInd, 1);

fprintf('--- Starting General Equilibrium Bisection ---\n');

while abs(diff) > tol && iter < max_iter 
    iter = iter + 1;
    R_guess = (R_max + R_min) / 2;
    
    % 2. Calculate the firm's rental rate of capital (r_k)
    r_k = R_guess - 1 + par.delta;
    
    % 3. Calculate Capital Demand and Wage using r_k
    k_d = (par.alpha / r_k)^(1 / (1 - par.alpha));
    w_guess = (1 - par.alpha) * k_d^par.alpha;

    par.R = R_guess;
    par.w = w_guess;

    solving_EGM = EGM(par, cpar, gri);
    kpol_k = solving_EGM.kpol_k;

% Create five 1D interpolants for the GE capital policy
    F_kpol_GE = cell(cpar.N, 1);
    for m = 1:cpar.N
        F_kpol_GE{m} = griddedInterpolant(gri.k, kpol_k(:, m), 'linear', 'linear');
    end

    k_i = zeros(numInd, time_GE);
    k_i(:, 1) = k_sim;

for t = 1:time_GE
        for j = 1:numInd
            if t < time_GE
                % Grab the discrete state from the z_idx matrix we made in Q1
                current_z_state = z_idx(j, t); 
                
                % Query directly over current k using the correct z-state interpolant
                k_prime = F_kpol_GE{current_z_state}(k_i(j,t));
                
                % REMOVED cpar.max: Enforce ONLY the borrowing constraint (0)
                k_i(j, t+1) = max(0, k_prime); 
            end
        end
    end

    k_sim = k_i(:, end);
    k_supply = mean(k_sim);
    diff = k_supply - k_d;

    fprintf('Iter %2d: R = %.4f | w = %.4f | K_sup = %.3f | K_dem = %.3f | Diff = %.3f\n', ...
             iter, R_guess, w_guess, k_supply, k_d, diff);

    if diff > 0 
        R_max = R_guess;
    else
        R_min = R_guess;
    end
end

fprintf('--- General Equilibrium Found! ---\n');
fprintf('Equilibrium Interest Rate R*: %.4f\n', R_guess);
fprintf('Equilibrium Wage w*: %.4f\n', w_guess);
fprintf('Aggregate Capital K*: %.4f\n', k_supply);

%% Plots 
figure('Name', 'GE Wealth Distribution');
hold on;
histogram(k_sim, 60, 'Normalization', 'pdf', 'FaceColor', [0.2 0.4 0.6], 'EdgeColor', 'w', 'DisplayName', 'Histogram');

[f_kde, xi_kde] = ksdensity(k_sim);
plot(xi_kde, f_kde, 'r-', 'LineWidth', 2, 'DisplayName', 'Kernel Density');

title('Stationary Wealth Distribution ($R^*$)', 'Interpreter', 'latex');
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Density', 'Interpreter', 'latex');
legend('Location', 'northeast');
grid on;
hold off;

%% Inequality Statistics
wealth_sorted = sort(k_sim);
N_ind = length(wealth_sorted);
gini_num = sum((1:N_ind)' .* wealth_sorted);
gini_den = N_ind * sum(wealth_sorted);
gini = (2 * gini_num / gini_den) - (N_ind + 1) / N_ind;

p1  = prctile(k_sim, 1);
p10 = prctile(k_sim, 10);
p90 = prctile(k_sim, 90);
p99 = prctile(k_sim, 99);

ratio_90_10 = p90 / p10;
ratio_99_1  = p99 / p1;

fprintf('\n--- Inequality Statistics (General Equilibrium) ---\n');
fprintf('Gini Coefficient:      %.4f\n', gini);
fprintf('90/10 Percentile Ratio: %.4f\n', ratio_90_10);
fprintf('99/1 Percentile Ratio:  %.4f\n', ratio_99_1);