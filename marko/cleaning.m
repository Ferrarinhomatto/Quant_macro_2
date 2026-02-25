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

% =========================================================================
%% Question 1.1: Environment Setup and Rouwenhorst Method
% =========================================================================
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

% Run Rouwenhorst and discretize z
% The Rouwenhorst method is robust for highly persistent processes (rho close to 1).
% It maps the continuous AR(1) productivity process to a discrete N-state Markov chain.
[gri.z, gri.prob] = rouwenhorst(cpar.N, cpar.mu, par.rho, sqrt(cpar.sigmae));

gri.z = exp(gri.z);
z_raw = gri.z;

[V, D] = eig(gri.prob');

[~, idx] = min(abs(diag(D) - 1));
pi_dist = V(:, idx) / sum(V(:, idx));

% Normalize the levels of z such that the mean of the stationary distribution is 1.
% This ensures aggregate labor supply in efficiency units equals 1 in equilibrium.
mean_z = pi_dist' * z_raw;
gri.z = z_raw / mean_z;

fprintf('--- Question 1.1: Rouwenhorst Method ---\n');
fprintf('Transition Matrix P:\n');
disp(gri.prob);
fprintf('Normalized Support z:\n');
disp(gri.z);
fprintf('Verified Mean (should be 1.0): %f\n', pi_dist' * gri.z);

% Pre-draw the discrete z_idx history for ALL simulations (Vectorized for Speed)
% Generating these beforehand ensures all setups test identical shock paths.
% It also avoids executing `rand` draws repeatedly inside the core loop.
numInd = 2000;
time = 1400;
z_idx = zeros(numInd, time);
z_idx(:, 1) = 3;

% Vectorized Markov chain simulation
rand_draws = rand(numInd, time - 1);
cum_prob = cumsum(gri.prob, 2);

for t = 1:time-1
    % Get cumulative probabilities for current states
    cumu_p = cum_prob(z_idx(:, t), :);
    % Compare uniform draws to cumulative probabilities
    z_idx(:, t+1) = sum(rand_draws(:, t) > cumu_p, 2) + 1;
end
z_i = gri.z(z_idx);


% =========================================================================
%% Question 1.2.A: Value Function Iteration (VFI)
% =========================================================================
% Set exactly what the Problem set statement asked for to answer Question 1
cpar.tol = 1e-6; 
cpar.Nkap = 20; 
cpar.min = 1e-6; 
cpar.max = 10;

gri.k = exp(linspace(log(cpar.min +1), log(cpar.max +1), cpar.Nkap)) - 1;

% === NEW EXTRAPOLATION TOGGLE FOR VFI ===
% Set to true for Setup 1 (allows agents to shoot past the k=10 grid)
% Set to false to strictly cap their choices at k=10
cpar.extrapolate = true; 

% Run VFI (Only once, since it is slow!)
fprintf('\n--- Question 1.2.A: VFI ---\n');
fprintf('Starting VFI...\n');
tic; 
[V, cpol, kpol] = VFI_GS(par, gri, cpar);
time_VFI = toc; 
fprintf('VFI Complete. Time taken: %.2f seconds\n', time_VFI);

% Plotting k'
figure('Name', 'VFI Policy Function k''');
hold on;
colors = lines(5); 
for j = 1:5
    plot(gri.k, kpol(j, :), 'LineWidth', 2, 'Color', colors(j, :), ...
         'DisplayName', sprintf('Productivity State $z_%d$', j));
end
plot(gri.k, gri.k, 'k--', 'LineWidth', 1.5, 'DisplayName', '45-degree line');
hold off;
title('Optimal Capital Policy Function: $k''(k,z)$', 'Interpreter', 'latex');
xlabel('Current Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Next Period Asset Holdings ($k''$)', 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'latex');
grid on;

% Plotting c
figure('Name', 'VFI Policy Function c');
hold on;
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

%% Simulate Panel and Calculate Mean Assets (VFI - Vectorized)
% Simulating 2000 agents. We use fast 1D interpolants (griddedInterpolant)
% to map continuous off-grid asset coordinates to the solved discrete policy functions.
c_i_VFI = zeros(numInd, time);
k_i_VFI = zeros(numInd, time);
k_i_VFI(:,1) = 5; % Initial capital

% Fast 1D Interpolants for VFI policies
F_cpol_VFI = cell(cpar.N, 1);
F_kpol_VFI = cell(cpar.N, 1);
for m = 1:cpar.N
    F_cpol_VFI{m} = griddedInterpolant(gri.k, cpol(m, :), 'linear', 'linear');
    F_kpol_VFI{m} = griddedInterpolant(gri.k, kpol(m, :), 'linear', 'linear');
end

% Vectorized Simulation
for t = 1:time
    for m = 1:cpar.N
        idx_m = (z_idx(:, t) == m);
        if any(idx_m)
            c_i_VFI(idx_m, t) = F_cpol_VFI{m}(k_i_VFI(idx_m, t));
            if t < time
                k_prime = F_kpol_VFI{m}(k_i_VFI(idx_m, t));
                % Enforce borrowing limit and strict max
                k_i_VFI(idx_m, t+1) = min(max(0, k_prime), cpar.max);
            end
        end
    end
end

% Discard the first 400 periods as requested
drop = 400;
k_steady_VFI = k_i_VFI(:, drop+1:end); 
k_vec_VFI = k_steady_VFI(:);

k_sorted_VFI = sort(k_vec_VFI);
N_k_VFI = length(k_sorted_VFI);
gini_VFI = (2 * sum((1:N_k_VFI)' .* k_sorted_VFI)) / (N_k_VFI * sum(k_sorted_VFI)) - (N_k_VFI + 1) / N_k_VFI;

fprintf('Mean Asset Holdings (discarding first %d periods): %.4f\n', drop, mean(k_vec_VFI));

%% Evaluate Euler Equation Errors (VFI)
c_implied = zeros(cpar.N, cpar.Nkap);
EE_VFI = zeros(cpar.N, cpar.Nkap);

for j = 1:cpar.N
    for k = 1:cpar.Nkap
        c_now = cpol(j, k);
        k_next = kpol(j, k);
        
        expected_MU = 0;
        for next_j = 1:cpar.N
            c_next = F_cpol_VFI{next_j}(k_next);
            if c_next <= 0
               c_next = 1e-10; 
            end
            prob_transition = gri.prob(j, next_j); 
            expected_MU = expected_MU + prob_transition * (c_next^(-par.sigma));
        end
        c_implied(j,k) = (par.beta * par.R * expected_MU)^(-1/par.sigma);
        EE_VFI(j,k) = abs((c_implied(j,k) - c_now) / c_now) * 100;
    end
end

fprintf('Maximum Euler Error for VFI (Absolute %%): %f%%\n', max(EE_VFI(:)));
max_EE_VFI1 = max(EE_VFI(:));

% Plot Wealth Distribution
figure('Name', 'VFI Wealth Distribution');
histogram(k_vec_VFI, 100, 'Normalization', 'pdf', 'FaceColor', [0.2 0.4 0.6], 'EdgeColor', 'none');
title('Stationary Wealth Distribution (VFI, $k_{max}=10$)', 'Interpreter', 'latex');
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Density', 'Interpreter', 'latex');
grid on;


% =========================================================================
%% Question 1.2.B: Endogenous-Gridpoints Method (EGM)
% =========================================================================
fprintf('\n--- Question 1.2.B: EGM ---\n');
tic;
solving_EGM = EGM(par, cpar, gri);
time_EGM = toc;
fprintf('EGM Complete. Time taken: %.2f seconds\n', time_EGM);
fprintf('EGM is %.2f times faster than VFI!\n', time_VFI / time_EGM);

% Compare VFI and EGM policies
max_pol_diff = max(abs(kpol' - solving_EGM.kpol_k), [], 'all');
fprintf('Max difference between VFI and EGM policy functions: %e\n', max_pol_diff);

% Compute Euler Equation Errors for EGM 1
% c(k,z) = R*k + w*z - k'(k,z)
cpol_EGM = zeros(cpar.N, cpar.Nkap);
for j = 1:cpar.N
    for k = 1:cpar.Nkap
        cpol_EGM(j,k) = par.R * gri.k(k) + par.w * gri.z(j) - solving_EGM.kpol_k(k, j);
    end
end

c_implied_EGM = zeros(cpar.N, cpar.Nkap);
EE_EGM = zeros(cpar.N, cpar.Nkap);

F_cpol_EGM1_c = cell(cpar.N, 1);
for m = 1:cpar.N
    F_cpol_EGM1_c{m} = griddedInterpolant(gri.k, cpol_EGM(m, :), 'linear', 'linear');
end

for j = 1:cpar.N
    for k = 1:cpar.Nkap
        c_now = cpol_EGM(j, k);
        k_next = solving_EGM.kpol_k(k, j);
        
        expected_MU = 0;
        for next_j = 1:cpar.N
            c_next = F_cpol_EGM1_c{next_j}(k_next);
            if c_next <= 0
               c_next = 1e-10; 
            end
            prob_transition = gri.prob(j, next_j); 
            expected_MU = expected_MU + prob_transition * (c_next^(-par.sigma));
        end
        c_implied_EGM(j,k) = (par.beta * par.R * expected_MU)^(-1/par.sigma);
        EE_EGM(j,k) = abs((c_implied_EGM(j,k) - c_now) / c_now) * 100;
    end
end
fprintf('Maximum Euler Error for EGM (Absolute %%): %f%%\n', max(EE_EGM(:)));
max_EE_EGM1 = max(EE_EGM(:));

% EGM Simulation (Vectorized)
k_i_EGM1 = zeros(numInd, time);
k_i_EGM1(:,1) = 5;

F_kpol_EGM1 = cell(cpar.N, 1);
for m = 1:cpar.N
    F_kpol_EGM1{m} = griddedInterpolant(gri.k, solving_EGM.kpol_k(:, m), 'linear', 'linear');
end

for t = 1:time
    for m = 1:cpar.N
        idx_m = (z_idx(:, t) == m);
        if any(idx_m) && t < time
            k_prime = F_kpol_EGM1{m}(k_i_EGM1(idx_m, t));
            % EXTRAPOLATION: Enforce ONLY the borrowing constraint (0)
            k_i_EGM1(idx_m, t+1) = max(0, k_prime);
        end
    end
end

k_vec_EGM1 = reshape(k_i_EGM1(:, drop+1:end), [], 1);
k_sorted_EGM1 = sort(k_vec_EGM1);
N_k_EGM1 = length(k_sorted_EGM1);
gini_EGM1 = (2 * sum((1:N_k_EGM1)' .* k_sorted_EGM1)) / (N_k_EGM1 * sum(k_sorted_EGM1)) - (N_k_EGM1 + 1) / N_k_EGM1;
mean_EGM1 = mean(k_vec_EGM1);


% =========================================================================
%% Question 2: General Equilibrium
% =========================================================================
R_min_GE = 1 - par.delta + 1e-4; 
R_max_GE = 1 / par.beta;
par.alpha = 0.33;

tol_GE = 1e-3;
diff_GE = tol_GE + 1;
iter_GE = 0;
max_iter_GE = 100;
time_GE = 300;
k_sim_GE1 = 5 * ones(numInd, 1); 

fprintf('\n--- Question 2: General Equilibrium Bisection ---\n');

% Bisection Search on the interest rate R:
% 1. Guess an interest rate R.
% 2. Calculate the implied wage w and firm's capital demand K_demand.
% 3. Solve the household's problem given these prices to find K_supply.
% 4. Adjust the interval bounds based on excess demand/supply of capital.
while abs(diff_GE) > tol_GE && iter_GE < max_iter_GE 
    iter_GE = iter_GE + 1;
    R_guess = (R_max_GE + R_min_GE) / 2;
    
    r_k = R_guess - 1 + par.delta;
    k_d = (par.alpha / r_k)^(1 / (1 - par.alpha));
    w_guess = (1 - par.alpha) * k_d^par.alpha;

    par.R = R_guess;
    par.w = w_guess;

    solving_EGM_GE = EGM(par, cpar, gri);
    kpol_k_GE = solving_EGM_GE.kpol_k;

    F_kpol_GE = cell(cpar.N, 1);
    for m = 1:cpar.N
        F_kpol_GE{m} = griddedInterpolant(gri.k, kpol_k_GE(:, m), 'linear', 'linear');
    end

    k_i_GE = zeros(numInd, time_GE);
    k_i_GE(:, 1) = k_sim_GE1;

    for t = 1:time_GE
        for m = 1:cpar.N
            idx_m = (z_idx(:, t) == m);
            if any(idx_m) && t < time_GE
                k_prime = F_kpol_GE{m}(k_i_GE(idx_m, t));
                k_i_GE(idx_m, t+1) = max(0, k_prime); 
            end
        end
    end

    k_sim_GE1 = k_i_GE(:, end);
    k_supply = mean(k_sim_GE1);
    diff_GE = k_supply - k_d;

    if diff_GE > 0 
        R_max_GE = R_guess;
    else
        R_min_GE = R_guess;
    end
end

R_star_1 = R_guess;
w_star_1 = w_guess;
K_star_1 = k_supply;

figure('Name', 'Question 2: GE Wealth Distribution');
hold on;
histogram(k_sim_GE1, 60, 'Normalization', 'pdf', 'FaceColor', [0.2 0.4 0.6], 'EdgeColor', 'w', 'DisplayName', 'Histogram');
bw1 = std(k_sim_GE1) / 3; % Heuristic for a wider, smoother bandwidth on right-skewed data
[f_kde, xi_kde] = ksdensity(k_sim_GE1, 'Bandwidth', bw1);
plot(xi_kde, f_kde, 'r-', 'LineWidth', 2, 'DisplayName', 'Kernel Density');
title('Stationary Wealth Distribution Setup 1 ($R^*, k_{max}=10$)', 'Interpreter', 'latex');
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Density', 'Interpreter', 'latex');
legend('Location', 'northeast');
grid on; hold off;

wealth_sorted_GE1 = sort(k_sim_GE1);
N_ind_GE1 = length(wealth_sorted_GE1);
gini_GE1 = (2 * sum((1:N_ind_GE1)' .* wealth_sorted_GE1) / (N_ind_GE1 * sum(wealth_sorted_GE1))) - (N_ind_GE1 + 1) / N_ind_GE1;

p1_GE1  = prctile(k_sim_GE1, 1);
p10_GE1 = prctile(k_sim_GE1, 10);
p90_GE1 = prctile(k_sim_GE1, 90);
p99_GE1 = prctile(k_sim_GE1, 99);
ratio_90_10_GE1 = p90_GE1 / p10_GE1;
ratio_99_1_GE1 = p99_GE1 / p1_GE1;


% =========================================================================
%% Robustness Check: Question 1 (Partial Equilibrium, k_max = 80)
% =========================================================================
% We examine the impact of relaxing the strict upper bound on assets.
% By setting k_max = 80 and disallowing extrapolation, we can capture
% the unconstrained dynamics of wealthy individuals who would otherwise overshoot.
fprintf('\n\n=== ROBUSTNESS CHECK Q1: k_max=80, NO EXTRAPOLATION ===\n');

cpar.max = 80; 
gri.k = exp(linspace(log(cpar.min + 1), log(cpar.max + 1), cpar.Nkap)) - 1;

par.w = 1; 
par.R = (1/par.beta) * 0.995; 
par.alpha = 0.33;
cpar.extrapolate = false; 

tic;
[~, cpol_PE2, kpol_PE2] = VFI_GS(par, gri, cpar);
time_VFI_PE2 = toc;

% Plotting k' (Robustness VFI)
figure('Name', 'VFI Policy Function k'' (Robustness, k_max=80)');
hold on;
colors = lines(5); 
for j = 1:5
    plot(gri.k, kpol_PE2(j, :), 'LineWidth', 2, 'Color', colors(j, :), ...
         'DisplayName', sprintf('Productivity State $z_%d$', j));
end
plot(gri.k, gri.k, 'k--', 'LineWidth', 1.5, 'DisplayName', '45-degree line');
hold off;
title('Optimal Capital Policy Function: $k''(k,z)$ ($k_{max}=80$)', 'Interpreter', 'latex');
xlabel('Current Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Next Period Asset Holdings ($k''$)', 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'latex');
grid on;

% Plotting c (Robustness VFI)
figure('Name', 'VFI Policy Function c (Robustness, k_max=80)');
hold on;
for j = 1:5
    plot(gri.k, cpol_PE2(j, :), 'LineWidth', 2, 'Color', colors(j, :), ...
         'DisplayName', sprintf('Productivity State $z_%d$', j));
end
title('Optimal Consumption Policy Function: $c(k,z)$ ($k_{max}=80$)', 'Interpreter', 'latex');
xlabel('Current Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Current Consumption ($c$)', 'Interpreter', 'latex');
legend('Location', 'northwest', 'Interpreter', 'latex');
grid on;
hold off;

tic;
solving_EGM_PE2 = EGM(par, cpar, gri);
kpol_k_PE2 = solving_EGM_PE2.kpol_k;
time_EGM_PE2 = toc;

% Euler Errors VFI Setup 2
c_implied2 = zeros(cpar.N, cpar.Nkap);
EE_VFI2 = zeros(cpar.N, cpar.Nkap);
F_cpol_VFI2_c = cell(cpar.N, 1);
for m = 1:cpar.N
    F_cpol_VFI2_c{m} = griddedInterpolant(gri.k, cpol_PE2(m, :), 'linear', 'linear');
end

for j = 1:cpar.N
    for k = 1:cpar.Nkap
        c_now = cpol_PE2(j, k);
        k_next = kpol_PE2(j, k);
        
        expected_MU = 0;
        for next_j = 1:cpar.N
            c_next = F_cpol_VFI2_c{next_j}(k_next);
            if c_next <= 0; c_next = 1e-10; end
            prob_transition = gri.prob(j, next_j); 
            expected_MU = expected_MU + prob_transition * (c_next^(-par.sigma));
        end
        c_implied2(j,k) = (par.beta * par.R * expected_MU)^(-1/par.sigma);
        EE_VFI2(j,k) = abs((c_implied2(j,k) - c_now) / c_now) * 100;
    end
end
max_EE_VFI2 = max(EE_VFI2(:));


% Euler Errors EGM Setup 2
cpol_EGM2 = zeros(cpar.N, cpar.Nkap);
for j = 1:cpar.N
    for k = 1:cpar.Nkap
        cpol_EGM2(j,k) = par.R * gri.k(k) + par.w * gri.z(j) - solving_EGM_PE2.kpol_k(k, j);
    end
end
c_implied_EGM2 = zeros(cpar.N, cpar.Nkap);
EE_EGM2 = zeros(cpar.N, cpar.Nkap);
F_cpol_EGM2_c = cell(cpar.N, 1);
for m = 1:cpar.N
    F_cpol_EGM2_c{m} = griddedInterpolant(gri.k, cpol_EGM2(m, :), 'linear', 'linear');
end
for j = 1:cpar.N
    for k = 1:cpar.Nkap
        c_now = cpol_EGM2(j, k);
        k_next = solving_EGM_PE2.kpol_k(k, j);
        
        expected_MU = 0;
        for next_j = 1:cpar.N
            c_next = F_cpol_EGM2_c{next_j}(k_next);
            if c_next <= 0; c_next = 1e-10; end
            prob_transition = gri.prob(j, next_j); 
            expected_MU = expected_MU + prob_transition * (c_next^(-par.sigma));
        end
        c_implied_EGM2(j,k) = (par.beta * par.R * expected_MU)^(-1/par.sigma);
        EE_EGM2(j,k) = abs((c_implied_EGM2(j,k) - c_now) / c_now) * 100;
    end
end
max_EE_EGM2 = max(EE_EGM2(:));

% Simulate PE Setup 2
k_i_VFI2 = zeros(numInd, time); k_i_VFI2(:,1) = 5;
k_i_EGM2 = zeros(numInd, time); k_i_EGM2(:,1) = 5;

F_kpol_VFI2 = cell(cpar.N, 1); F_kpol_EGM2 = cell(cpar.N, 1);
for m = 1:cpar.N
    F_kpol_VFI2{m} = griddedInterpolant(gri.k, kpol_PE2(m, :), 'linear', 'linear');
    F_kpol_EGM2{m} = griddedInterpolant(gri.k, kpol_k_PE2(:, m), 'linear', 'linear');
end

for t = 1:time
    for m = 1:cpar.N
        idx_m = (z_idx(:, t) == m);
        if any(idx_m) && t < time
             k_prime_VFI = F_kpol_VFI2{m}(k_i_VFI2(idx_m, t));
             k_i_VFI2(idx_m, t+1) = min(max(0, k_prime_VFI), cpar.max); 
             
             k_prime_EGM = F_kpol_EGM2{m}(k_i_EGM2(idx_m, t));
             k_i_EGM2(idx_m, t+1) = min(max(0, k_prime_EGM), cpar.max); 
        end
    end
end

k_vec_VFI2 = reshape(k_i_VFI2(:, drop+1:end), [], 1);
k_sorted_VFI2 = sort(k_vec_VFI2);
N_k_VFI2 = length(k_sorted_VFI2);
gini_VFI2 = (2 * sum((1:N_k_VFI2)' .* k_sorted_VFI2)) / (N_k_VFI2 * sum(k_sorted_VFI2)) - (N_k_VFI2 + 1) / N_k_VFI2;
mean_VFI2 = mean(k_vec_VFI2);

k_vec_EGM2 = reshape(k_i_EGM2(:, drop+1:end), [], 1);
k_sorted_EGM2 = sort(k_vec_EGM2);
N_k_EGM2 = length(k_sorted_EGM2);
gini_EGM2 = (2 * sum((1:N_k_EGM2)' .* k_sorted_EGM2)) / (N_k_EGM2 * sum(k_sorted_EGM2)) - (N_k_EGM2 + 1) / N_k_EGM2;
mean_EGM2 = mean(k_vec_EGM2);

% Plot Wealth Distribution for VFI Robustness
figure('Name', 'VFI Wealth Distribution (Robustness, k_max=80)');
histogram(k_vec_VFI2, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.2 0.6], 'EdgeColor', 'none');
title('Stationary Wealth Distribution (VFI Robustness, $k_{max}=80$)', 'Interpreter', 'latex');
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Density', 'Interpreter', 'latex');
grid on;


% =========================================================================
%% Robustness Check: Question 2 (General Equilibrium, k_max = 80)
% =========================================================================
R_min_GE2 = 1 - par.delta + 1e-4; R_max_GE2 = 1 / par.beta;
diff_GE2 = 1; iter_GE2 = 0;
k_sim_GE2 = 5 * ones(numInd, 1); 

while abs(diff_GE2) > tol_GE && iter_GE2 < max_iter_GE 
    iter_GE2 = iter_GE2 + 1;
    R_guess = (R_max_GE2 + R_min_GE2) / 2;
    r_k = R_guess - 1 + par.delta;
    k_d = (par.alpha / r_k)^(1 / (1 - par.alpha));
    par.w = (1 - par.alpha) * k_d^par.alpha; par.R = R_guess;

    solving_EGM_GE2 = EGM(par, cpar, gri);
    F_kpol_GE2 = cell(cpar.N, 1);
    for m = 1:cpar.N
        F_kpol_GE2{m} = griddedInterpolant(gri.k, solving_EGM_GE2.kpol_k(:, m), 'linear', 'linear');
    end

    k_i_GE2 = zeros(numInd, time_GE); k_i_GE2(:, 1) = k_sim_GE2;
    for t = 1:time_GE
        for m = 1:cpar.N
            idx_m = (z_idx(:, t) == m);
            if any(idx_m) && t < time_GE
                k_prime = F_kpol_GE2{m}(k_i_GE2(idx_m, t));
                k_i_GE2(idx_m, t+1) = min(max(0, k_prime), cpar.max); 
            end
        end
    end
    k_sim_GE2 = k_i_GE2(:, end);
    k_supply = mean(k_sim_GE2); diff_GE2 = k_supply - k_d;
    if diff_GE2 > 0; R_max_GE2 = R_guess; else; R_min_GE2 = R_guess; end
end

R_star_2 = R_guess;
w_star_2 = par.w;
K_star_2 = k_supply;

figure('Name', 'Question 2 (Robustness): GE Wealth Distribution');
hold on;
histogram(k_sim_GE2, 60, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.4], 'EdgeColor', 'w', 'DisplayName', 'Histogram');
bw2 = std(k_sim_GE2) / 3; % Heuristic for a wider, smoother bandwidth on right-skewed data
[f_kde, xi_kde] = ksdensity(k_sim_GE2, 'Bandwidth', bw2);
plot(xi_kde, f_kde, 'r-', 'LineWidth', 2, 'DisplayName', 'Kernel Density');
title('Stationary Wealth Distribution Setup 2 ($R^*, k_{max}=80$)', 'Interpreter', 'latex');
xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex');
ylabel('Density', 'Interpreter', 'latex');
legend('Location', 'northeast');
grid on; hold off;

wealth_sorted_GE2 = sort(k_sim_GE2);
N_ind_GE2 = length(wealth_sorted_GE2);
gini_GE2 = (2 * sum((1:N_ind_GE2)' .* wealth_sorted_GE2) / (N_ind_GE2 * sum(wealth_sorted_GE2))) - (N_ind_GE2 + 1) / N_ind_GE2;
p1_GE2  = prctile(k_sim_GE2, 1);
p10_GE2 = prctile(k_sim_GE2, 10);
p90_GE2 = prctile(k_sim_GE2, 90);
p99_GE2 = prctile(k_sim_GE2, 99);
ratio_90_10_GE2 = p90_GE2 / p10_GE2;
ratio_99_1_GE2  = p99_GE2 / p1_GE2;

%% ---------------------------------------------------------------------------------
% FINAL COMPREHENSIVE TABLES
% ---------------------------------------------------------------------------------
fprintf('\n===================================================================================================\n');
fprintf('TABLE 1: PARTIAL EQUILIBRIUM PERFORMANCE & STATISTICS\n');
fprintf('===================================================================================================\n');
fprintf('%-30s | %-12s | %-16s | %-15s | %-15s\n', 'Model Setup', 'Runtime (s)', 'Max Euler Er (%)', 'Mean Assets', 'Gini Coeff');
fprintf('---------------------------------------------------------------------------------------------------\n');
fprintf('%-30s | %-12.2f | %-16.6f | %-15.4f | %-15.4f\n', '1. Baseline VFI (k_max=10)', time_VFI, max_EE_VFI1, mean(k_vec_VFI), gini_VFI);
fprintf('%-30s | %-12.2f | %-16.6f | %-15.4f | %-15.4f\n', '2. Baseline EGM (k_max=10)', time_EGM, max_EE_EGM1, mean_EGM1, gini_EGM1);
fprintf('%-30s | %-12.2f | %-16.6f | %-15.4f | %-15.4f\n', '3. Robust VFI (k_max=80)', time_VFI_PE2, max_EE_VFI2, mean_VFI2, gini_VFI2);
fprintf('%-30s | %-12.2f | %-16.6f | %-15.4f | %-15.4f\n', '4. Robust EGM (k_max=80)', time_EGM_PE2, max_EE_EGM2, mean_EGM2, gini_EGM2);
fprintf('===================================================================================================\n');

fprintf('\n===================================================================================================\n');
fprintf('TABLE 2: GENERAL EQUILIBRIUM RESULTS\n');
fprintf('===================================================================================================\n');
fprintf('%-25s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n', 'Setup', 'Equil R*', 'Equil w*', 'Agg K*', 'Gini', 'P90/P10', 'P99/P1');
fprintf('---------------------------------------------------------------------------------------------------\n');
fprintf('%-25s | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', 'Q2 Baseline (k_max=10)', R_star_1, w_star_1, K_star_1, gini_GE1, ratio_90_10_GE1, ratio_99_1_GE1);
fprintf('%-25s | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', 'Q2 Robust (k_max=80)', R_star_2, w_star_2, K_star_2, gini_GE2, ratio_90_10_GE2, ratio_99_1_GE2);
fprintf('===================================================================================================\n');

% =========================================================================
%% Question 3: Heterogeneous Asset Returns
% =========================================================================
fprintf('\n=================================================================================\n');
fprintf('--- Question 3: Heterogeneous Asset Returns ---\n');
fprintf('=================================================================================\n');

% Define the 3 parameter setups for consistent comparison
setups = struct('name', {'Setup 1 (k\_max=10, No Extrap)', 'Setup 2 (k\_max=10, Extrap)', 'Setup 3 (k\_max=80, No Extrap)'}, ...
                'k_max', {10, 10, 80}, ...
                'extrapolate', {false, true, false});

% Solver config for Heterogeneous Returns
cpar.N_het = 25;
cpar.maxit = 100;
par.alpha = 0.33;
par.w = 1;

% Arrays to store Results for the final table

Q3_results = struct('r_bar', zeros(3, 2), 'w', zeros(3, 2), 'K', zeros(3, 2), 'Gini', zeros(3, 2), 'p90_10', zeros(3, 2), 'p99_1', zeros(3, 2));

rho_r_cases = [0, 0.9];
colors_r = {'b', 'r'};

for s_idx = 1:numel(setups)
    current_setup = setups(s_idx);
    fprintf('\n\n=================================================================================\n');
    fprintf('--- Question 3 Setup %d: %s ---\n', s_idx, current_setup.name);
    fprintf('=================================================================================\n');
    
    % Re-initialize grids per setup
    cpar.max = current_setup.k_max;
    gri.k = exp(linspace(log(cpar.min + 1), log(cpar.max + 1), cpar.Nkap)) - 1;
    
    % Prepare the Pareto Plot for THIS specific setup
    f_pareto = figure('Name', sprintf('Q3: Pareto Tail - %s', current_setup.name), 'NumberTitle', 'on');
    hold on;

for case_idx = 1:2
    rho_r = rho_r_cases(case_idx);
    fprintf('\n--- Solving Case %d: rho_r = %.1f ---\n', case_idx, rho_r);
    
    % 1. Discretize the excess return process
    sigma_r2 = (1 - rho_r^2) * 0.002;
    [r_grid, P_r] = rouwenhorst(5, 0, rho_r, sqrt(sigma_r2));
    
    % 2. Construct the 25x25 Kronecker Space
    % We assume independent processes: P(z', r' | z, r) = P(z'|z) * P(r'|r)
    gri_het.prob = kron(gri.prob, P_r);
    
    % Use kron to stretch the 5x1 grids into 25x1 arrays mapping to the transition matrix
    % Because P_r is the INNER matrix, the `r` states change faster than the `z` states
    gri_het.z = kron(gri.z, ones(5, 1));
    gri_het.r = kron(ones(5, 1), r_grid);
    gri_het.k = gri.k;
    
    % Pre-simulate the independent r_idx Markov Chain (Vectorized)
    r_idx = zeros(numInd, time);
    r_idx(:, 1) = 3;
    rand_draws_r = rand(numInd, time - 1);
    cum_prob_r = cumsum(P_r, 2);
    for t = 1:time-1
        cumu_p_r = cum_prob_r(r_idx(:, t), :);
        r_idx(:, t+1) = sum(rand_draws_r(:, t) > cumu_p_r, 2) + 1;
    end
    
    % Need a matching length arrays for the General Equilibrium phase (time_GE instead of time)
    z_idx_GE = z_idx(:, 1:time_GE);
    r_idx_GE = r_idx(:, 1:time_GE);
    
    % Combine to a unified index 1 through 25
    % Since r_grid is the inner Kronecker product (rapidly varying), 
    % the formula for a combined 2D index is: (outer_idx - 1) * N_inner + inner_idx
    het_idx = (z_idx_GE - 1) * 5 + r_idx_GE;
    
    %% Q3.1: Solve Partial Equilibrium for given r_bar
    par.r_bar = 0.985 / par.beta - 1;
    fprintf('Solving PE EGM for given r_bar = %.4f...\n', par.r_bar);
    solving_EGM_het = EGM_het(par, cpar, gri_het);
    
    % Plot policy functions for highest/lowest return states
    figure('Name', sprintf('Q3 Policies (Setup %d, rho_r=%.1f)', s_idx, rho_r));
    plot(gri.k, solving_EGM_het.kpol_k(:, 21), 'r-', 'LineWidth', 2, 'DisplayName', 'Highest z, Lowest r'); hold on;
    plot(gri.k, solving_EGM_het.kpol_k(:, 25), 'b-', 'LineWidth', 2, 'DisplayName', 'Highest z, Highest r');
    plot(gri.k, gri.k, 'k--', 'HandleVisibility', 'off');
    title(sprintf('Savings Policy (Setup %d, $\\rho^r = %.1f$)', s_idx, rho_r), 'Interpreter', 'latex');
    xlabel('Assets ($k$)', 'Interpreter', 'latex'); ylabel('Next Assets ($k''$)', 'Interpreter', 'latex');
    legend('Location', 'northwest'); grid on; hold off;

    %% Q3.2: General Equilibrium Bisection for r_bar
    r_bar_min = par.delta - 0.02; % Floor near zero return
    r_bar_max = par.r_bar + 0.05; 
    tol_GE = 1e-3; diff_GE = 1; iter_GE = 0;
    
    k_sim_GE_het = 5 * ones(numInd, 1);
    
    fprintf('Starting General Equilibrium Bisection...\n');
    while abs(diff_GE) > tol_GE && iter_GE < 50
        iter_GE = iter_GE + 1;
        par.r_bar = (r_bar_max + r_bar_min) / 2;
        
        % Firm FOCs (Aggregate R is 1 + r_bar, so r_k = r_bar + delta)
        r_k = par.r_bar + par.delta;
        
        % Check if r_k is valid
        if r_k <= 0
            fprintf('Warning: Invalid r_k = %.4f detected. Adjusting bounds.\n', r_k);
            r_bar_min = par.r_bar;
            continue;
        end
        
        k_d = (par.alpha / r_k)^(1 / (1 - par.alpha));
        par.w = (1 - par.alpha) * k_d^par.alpha;
        
        solving_GE_het = EGM_het(par, cpar, gri_het);
        
        F_kpol_het = cell(cpar.N_het, 1);
        for m = 1:cpar.N_het
            F_kpol_het{m} = griddedInterpolant(gri.k, solving_GE_het.kpol_k(:, m), 'linear', 'linear');
        end
        
        k_i_het = zeros(numInd, time_GE);
        k_i_het(:, 1) = k_sim_GE_het;
        for t = 1:time_GE
            for m = 1:cpar.N_het
                idx_m = (het_idx(:, t) == m);
                if any(idx_m) && t < time_GE
                    k_prime = F_kpol_het{m}(k_i_het(idx_m, t));
                    % Apply interpolation constraint dynamically based on setup
                    if current_setup.extrapolate
                        k_i_het(idx_m, t+1) = max(0, k_prime); 
                    else
                        k_i_het(idx_m, t+1) = min(max(0, k_prime), cpar.max); 
                    end
                end
            end
        end
        
        k_sim_GE_het = k_i_het(:, end);
        k_supply = mean(k_sim_GE_het);
        diff_GE = k_supply - k_d;
        
        fprintf('Iter %2d: r_bar=%.4f | w=%.4f | K_sup=%.3f | K_dem=%.3f | Diff=%.3f\n', ...
            iter_GE, par.r_bar, par.w, k_supply, k_d, diff_GE);
            
        if diff_GE > 0; r_bar_max = par.r_bar; else; r_bar_min = par.r_bar; end
    end
    
    %% Q3.3 Inequality Stats and Pareto Tail
    wealth_sorted_het = sort(k_sim_GE_het);
    N_ind_het = length(wealth_sorted_het);
    gini_het = (2*sum((1:N_ind_het)'.*wealth_sorted_het)/(N_ind_het*sum(wealth_sorted_het))) - (N_ind_het+1)/N_ind_het;
    
    p1_het  = prctile(k_sim_GE_het, 1);
    p10_het = prctile(k_sim_GE_het, 10);
    p90_het = prctile(k_sim_GE_het, 90);
    p99_het = prctile(k_sim_GE_het, 99);
    ratio_90_10_het = p90_het / p10_het;
    ratio_99_1_het  = p99_het / p1_het;
    
    % Store Results
    Q3_results.r_bar(s_idx, case_idx) = par.r_bar;
    Q3_results.w(s_idx, case_idx) = par.w;
    Q3_results.K(s_idx, case_idx) = k_supply;
    Q3_results.Gini(s_idx, case_idx) = gini_het;
    Q3_results.p90_10(s_idx, case_idx) = ratio_90_10_het;
    Q3_results.p99_1(s_idx, case_idx) = ratio_99_1_het;
    
    % Plot Histogram and Kernel Density
    figure('Name', sprintf('Q3 Wealth Distribution (Setup %d, rho_r=%.1f)', s_idx, rho_r));
    hold on;
    histogram(k_sim_GE_het, 60, 'Normalization', 'pdf', 'FaceColor', [0.3 0.5 0.7], 'EdgeColor', 'w', 'DisplayName', 'Histogram');
    bw_het = std(k_sim_GE_het) / 2; % Heuristic for smoother bandwidth
    [f_kde_het, xi_kde_het] = ksdensity(k_sim_GE_het, 'Bandwidth', bw_het);
    plot(xi_kde_het, f_kde_het, 'r-', 'LineWidth', 2, 'DisplayName', 'Kernel Density');
    title(sprintf('Stationary Wealth Distribution (Setup %d, $\\rho^r = %.1f$)', s_idx, rho_r), 'Interpreter', 'latex');
    xlabel('Asset Holdings ($k$)', 'Interpreter', 'latex');
    ylabel('Density', 'Interpreter', 'latex');
    legend('Location', 'northeast');
    grid on; hold off;
    
    % Pareto Tail (Log-Log Survival Function)
    % S(k) = 1 - CDF(k)
    empirical_cdf = (1:N_ind_het)' / N_ind_het;
    survival_func = 1 - empirical_cdf;
    
    % Drop the absolute zero/boundary values for clean log calculation
    valid_idx = wealth_sorted_het > 0.1 & survival_func > 0;
    
    figure(f_pareto); % Bring Pareto figure forward
    plot(log(wealth_sorted_het(valid_idx)), log(survival_func(valid_idx)), ...
        colors_r{case_idx}, 'LineWidth', 2, 'DisplayName', sprintf('$\\rho^r = %.1f$', rho_r));
end

figure(f_pareto);
title(sprintf('Pareto Tail: %s', current_setup.name), 'Interpreter', 'latex');
xlabel('$\log(k)$', 'Interpreter', 'latex');
ylabel('$\log(1 - \mathrm{CDF}(k))$', 'Interpreter', 'latex');
legend('Location', 'southwest', 'Interpreter', 'latex');
grid on; hold off;

end % End outer setup loop

%% ---------------------------------------------------------------------------------
% COMPREHENSIVE FINAL COMPARISON TABLE
% ---------------------------------------------------------------------------------
fprintf('\n==========================================================================================================================\n');
fprintf('TABLE 3: COMPREHENSIVE GENERAL EQUILIBRIUM COMPARISON (Representative vs. Heterogeneous Returns)\n');
fprintf('==========================================================================================================================\n');
fprintf('%-30s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s\n', 'Framework \ Setup', 'Equil r*', 'Equil w*', 'Agg K*', 'Gini', 'P90/P10', 'P99/P1');
fprintf('--------------------------------------------------------------------------------------------------------------------------\n');
fprintf('%-30s | %-12.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', 'Q2 Setup 1 (Rep, k_max=10)', R_star_1 - 1, w_star_1, K_star_1, gini_GE1, ratio_90_10_GE1, ratio_99_1_GE1);
fprintf('%-30s | %-12.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', 'Q2 Setup 3 (Rep, k_max=80)', R_star_2 - 1, w_star_2, K_star_2, gini_GE2, ratio_90_10_GE2, ratio_99_1_GE2);
fprintf('--------------------------------------------------------------------------------------------------------------------------\n');

for s_idx = 1:numel(setups)
    for case_idx = 1:2
        % Convert inner loop index back to rho_r label
        rho_r_lbl = rho_r_cases(case_idx);
        sys_name = sprintf('Q3 %s (rho=%.1f)', setups(s_idx).name, rho_r_lbl);
        fprintf('%-30s | %-12.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', ...
            sys_name, Q3_results.r_bar(s_idx, case_idx), Q3_results.w(s_idx, case_idx), ...
            Q3_results.K(s_idx, case_idx), Q3_results.Gini(s_idx, case_idx), ...
            Q3_results.p90_10(s_idx, case_idx), Q3_results.p99_1(s_idx, case_idx));
    end
end
fprintf('==========================================================================================================================\n');