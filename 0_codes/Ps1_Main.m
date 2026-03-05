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

rng(1);
addpath("0_functions/");

% =========================================================================
% === GLOBAL TOGGLE FOR ROBUSTNESS CHECKS ===
% Set to true to run all models (including k_max=200). Set to false for baseline only.
run_robustness = true;
% Set to false to skip the slow VFI re-run in robustness (saves ~2-5 min).
run_VFI_robustness = true;

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
cpar.Nkap = 40; 
cpar.min = 0; 
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

% (Q1 VFI k' policy — see combined figure below)
colors = lines(5);
if ~exist('1_graphs', 'dir'); mkdir('1_graphs'); end

% (Q1 VFI c policy — see combined figure below)

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
EE_VFI = euler_errors(par.beta, par.sigma, par.R, par.w, kpol, gri.k(:), gri.z(:), gri.prob);

interior_mask_VFI1 = kpol > 1e-5;
fprintf('Maximum Interior Euler Error for VFI (Absolute %%): %e%%\n', max(EE_VFI(interior_mask_VFI1)));
max_EE_VFI1 = max(EE_VFI(interior_mask_VFI1));

% (Q1 VFI wealth dist + Euler errors — see combined figure below)


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

% (Q1 EGM k' policy — see combined figure below)

% Compute c for EGM plots and errors
cpol_EGM = zeros(cpar.N, cpar.Nkap);
for j = 1:cpar.N
    for k = 1:cpar.Nkap
        cpol_EGM(j,k) = par.R * gri.k(k) + par.w * gri.z(j) - solving_EGM.kpol_k(k, j);
    end
end

% (Q1 EGM c policy — see combined figure below)


% Compute Euler Equation Errors for EGM 1
EE_EGM = euler_errors(par.beta, par.sigma, par.R, par.w, solving_EGM.kpol_k', gri.k(:), gri.z(:), gri.prob);

interior_mask_EGM1 = solving_EGM.kpol_k' > 1e-5;
fprintf('Maximum Interior Euler Error for EGM (Absolute %%): %e%%\n', max(EE_EGM(interior_mask_EGM1)));
max_EE_EGM1 = max(EE_EGM(interior_mask_EGM1));

% (Q1 EGM Euler errors — see combined figure below)

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

% Save baseline grid before robustness overwrites it
gri_k_base = gri.k;


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

% (Q2 GE baseline wealth dist — see combined figure below)

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
if run_robustness
% We examine the impact of relaxing the strict upper bound on assets.
% By setting k_max = 80 and disallowing extrapolation, we can capture
% the unconstrained dynamics of wealthy individuals who would otherwise overshoot.
fprintf('\n\n=== ROBUSTNESS CHECK Q1: k_max=200, NO EXTRAPOLATION ===\n');

cpar.max = 200; 
gri.k = exp(linspace(log(cpar.min + 1), log(cpar.max + 1), cpar.Nkap)) - 1;

par.w = 1; 
par.R = (1/par.beta) * 0.995; 
par.alpha = 0.33;
cpar.extrapolate = false; 

if run_VFI_robustness
tic;
[~, cpol_PE2, kpol_PE2] = VFI_GS(par, gri, cpar);
time_VFI_PE2 = toc;

% (Q1 VFI robustness plots — see combined figure below)
end % end run_VFI_robustness guard for VFI plots

tic;
solving_EGM_PE2 = EGM(par, cpar, gri);
kpol_k_PE2 = solving_EGM_PE2.kpol_k;
time_EGM_PE2 = toc;

% Euler Errors for Robustness
if run_VFI_robustness
    EE_VFI2 = euler_errors(par.beta, par.sigma, par.R, par.w, kpol_PE2, gri.k(:), gri.z(:), gri.prob);
    interior_mask_VFI2 = kpol_PE2 > 1e-5;
    max_EE_VFI2 = max(EE_VFI2(interior_mask_VFI2));
end

EE_EGM2 = euler_errors(par.beta, par.sigma, par.R, par.w, solving_EGM_PE2.kpol_k', gri.k(:), gri.z(:), gri.prob);
interior_mask_EGM2 = solving_EGM_PE2.kpol_k' > 1e-5;
max_EE_EGM2 = max(EE_EGM2(interior_mask_EGM2));

% Simulate PE Setup 2
if run_VFI_robustness
k_i_VFI2 = zeros(numInd, time); k_i_VFI2(:,1) = 5;

F_kpol_VFI2 = cell(cpar.N, 1);
for m = 1:cpar.N
    F_kpol_VFI2{m} = griddedInterpolant(gri.k, kpol_PE2(m, :), 'linear', 'linear');
end
end % end run_VFI_robustness guard

k_i_EGM2 = zeros(numInd, time); k_i_EGM2(:,1) = 5;
F_kpol_EGM2 = cell(cpar.N, 1);
for m = 1:cpar.N
    F_kpol_EGM2{m} = griddedInterpolant(gri.k, kpol_k_PE2(:, m), 'linear', 'linear');
end

for t = 1:time
    for m = 1:cpar.N
        idx_m = (z_idx(:, t) == m);
        if any(idx_m) && t < time
             if run_VFI_robustness
                 k_prime_VFI = F_kpol_VFI2{m}(k_i_VFI2(idx_m, t));
                 k_i_VFI2(idx_m, t+1) = min(max(0, k_prime_VFI), cpar.max);
             end
             
             k_prime_EGM = F_kpol_EGM2{m}(k_i_EGM2(idx_m, t));
             k_i_EGM2(idx_m, t+1) = min(max(0, k_prime_EGM), cpar.max); 
        end
    end
end

if run_VFI_robustness
k_vec_VFI2 = reshape(k_i_VFI2(:, drop+1:end), [], 1);
k_sorted_VFI2 = sort(k_vec_VFI2);
N_k_VFI2 = length(k_sorted_VFI2);
gini_VFI2 = (2 * sum((1:N_k_VFI2)' .* k_sorted_VFI2)) / (N_k_VFI2 * sum(k_sorted_VFI2)) - (N_k_VFI2 + 1) / N_k_VFI2;
mean_VFI2 = mean(k_vec_VFI2);
end

k_vec_EGM2 = reshape(k_i_EGM2(:, drop+1:end), [], 1);
k_sorted_EGM2 = sort(k_vec_EGM2);
N_k_EGM2 = length(k_sorted_EGM2);
gini_EGM2 = (2 * sum((1:N_k_EGM2)' .* k_sorted_EGM2)) / (N_k_EGM2 * sum(k_sorted_EGM2)) - (N_k_EGM2 + 1) / N_k_EGM2;
mean_EGM2 = mean(k_vec_EGM2);


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

% (Q2 GE robustness wealth dist — see combined figure below)

% Save robustness grid
gri_k_rob = gri.k;

wealth_sorted_GE2 = sort(k_sim_GE2);
N_ind_GE2 = length(wealth_sorted_GE2);
gini_GE2 = (2 * sum((1:N_ind_GE2)' .* wealth_sorted_GE2) / (N_ind_GE2 * sum(wealth_sorted_GE2))) - (N_ind_GE2 + 1) / N_ind_GE2;
p1_GE2  = prctile(k_sim_GE2, 1);
p10_GE2 = prctile(k_sim_GE2, 10);
p90_GE2 = prctile(k_sim_GE2, 90);
p99_GE2 = prctile(k_sim_GE2, 99);
ratio_90_10_GE2 = p90_GE2 / p10_GE2;
ratio_99_1_GE2  = p99_GE2 / p1_GE2;
end

%% =========================================================================
% COMBINED Q1 PE PLOTS
% =========================================================================
colors = lines(5);

% --- Figure 1: Combined k' Policy Functions (2x2) ---
fig_kpol = figure('Name', 'Q1 Combined: Capital Policy Functions');
fig_kpol.Position = [50, 50, 1200, 800];

subplot(2,2,1); hold on;
for j = 1:5
    plot(gri_k_base, kpol(j,:), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
end
plot(gri_k_base, gri_k_base, 'k--', 'LineWidth', 1.5, 'DisplayName', '45$^\circ$');
title('VFI ($k_{max}=10$)', 'Interpreter', 'latex');
xlabel('$k$', 'Interpreter', 'latex'); ylabel('$k''$', 'Interpreter', 'latex');
legend('Location','northwest','Interpreter','latex','FontSize',6); grid on; hold off;

subplot(2,2,2); hold on;
for j = 1:5
    plot(gri_k_base, solving_EGM.kpol_k(:,j), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
end
plot(gri_k_base, gri_k_base, 'k--', 'LineWidth', 1.5, 'DisplayName', '45$^\circ$');
title('EGM ($k_{max}=10$)', 'Interpreter', 'latex');
xlabel('$k$', 'Interpreter', 'latex'); ylabel('$k''$', 'Interpreter', 'latex');
legend('Location','northwest','Interpreter','latex','FontSize',6); grid on; hold off;

if run_robustness
    if run_VFI_robustness
        subplot(2,2,3); hold on;
        for j = 1:5
            plot(gri_k_rob, kpol_PE2(j,:), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
        end
        plot(gri_k_rob, gri_k_rob, 'k--', 'LineWidth', 1.5, 'DisplayName', '45$^\circ$');
        title('VFI ($k_{max}=200$)', 'Interpreter', 'latex');
        xlabel('$k$', 'Interpreter', 'latex'); ylabel('$k''$', 'Interpreter', 'latex');
        legend('Location','northwest','Interpreter','latex','FontSize',6); grid on; hold off;
    end
    subplot(2,2,4); hold on;
    for j = 1:5
        plot(gri_k_rob, kpol_k_PE2(:,j), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
    end
    plot(gri_k_rob, gri_k_rob, 'k--', 'LineWidth', 1.5, 'DisplayName', '45$^\circ$');
    title('EGM ($k_{max}=200$)', 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('$k''$', 'Interpreter', 'latex');
    legend('Location','northwest','Interpreter','latex','FontSize',6); grid on; hold off;
end
sgtitle('Q1: Capital Policy Functions $k''(k,z)$', 'Interpreter', 'latex', 'FontSize', 14);


% --- Figure 2: Combined c Policy Functions (2x2) ---
fig_cpol = figure('Name', 'Q1 Combined: Consumption Policy Functions');
fig_cpol.Position = [50, 50, 1200, 800];

subplot(2,2,1); hold on;
for j = 1:5
    plot(gri_k_base, cpol(j,:), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
end
title('VFI ($k_{max}=10$)', 'Interpreter', 'latex');
xlabel('$k$', 'Interpreter', 'latex'); ylabel('$c$', 'Interpreter', 'latex');
legend('Location','northwest','Interpreter','latex','FontSize',6); grid on; hold off;

subplot(2,2,2); hold on;
for j = 1:5
    plot(gri_k_base, cpol_EGM(j,:), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
end
title('EGM ($k_{max}=10$)', 'Interpreter', 'latex');
xlabel('$k$', 'Interpreter', 'latex'); ylabel('$c$', 'Interpreter', 'latex');
legend('Location','northwest','Interpreter','latex','FontSize',6); grid on; hold off;

if run_robustness && run_VFI_robustness
    subplot(2,2,3); hold on;
    for j = 1:5
        plot(gri_k_rob, cpol_PE2(j,:), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
    end
    title('VFI ($k_{max}=200$)', 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('$c$', 'Interpreter', 'latex');
    legend('Location','northwest','Interpreter','latex','FontSize',6); grid on; hold off;
end
sgtitle('Q1: Consumption Policy Functions $c(k,z)$', 'Interpreter', 'latex', 'FontSize', 14);


% --- Figure 3: Combined Euler Equation Errors (2x2) ---
fig_ee = figure('Name', 'Q1 Combined: Euler Equation Errors');
fig_ee.Position = [50, 50, 1200, 800];

subplot(2,2,1); hold on;
for j = 1:5
    vp = kpol(j,:) > 1e-5;
    plot(gri_k_base(vp), log10(EE_VFI(j,vp)/100), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
end
title('VFI ($k_{max}=10$)', 'Interpreter', 'latex');
xlabel('$k$', 'Interpreter', 'latex'); ylabel('$\log_{10}$ Error', 'Interpreter', 'latex');
legend('Location','northeast','Interpreter','latex','FontSize',6); grid on; hold off;

subplot(2,2,2); hold on;
for j = 1:5
    vp = solving_EGM.kpol_k(:,j) > 1e-5;
    plot(gri_k_base(vp), log10(EE_EGM(j,vp)/100), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
end
title('EGM ($k_{max}=10$)', 'Interpreter', 'latex');
xlabel('$k$', 'Interpreter', 'latex'); ylabel('$\log_{10}$ Error', 'Interpreter', 'latex');
legend('Location','northeast','Interpreter','latex','FontSize',6); grid on; hold off;

if run_robustness
    if run_VFI_robustness
        subplot(2,2,3); hold on;
        for j = 1:5
            vp = kpol_PE2(j,:) > 1e-5;
            plot(gri_k_rob(vp), log10(EE_VFI2(j,vp)/100), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
        end
        title('VFI ($k_{max}=200$)', 'Interpreter', 'latex');
        xlabel('$k$', 'Interpreter', 'latex'); ylabel('$\log_{10}$ Error', 'Interpreter', 'latex');
        legend('Location','northeast','Interpreter','latex','FontSize',6); grid on; hold off;
    end
    subplot(2,2,4); hold on;
    for j = 1:5
        vp = solving_EGM_PE2.kpol_k(:,j) > 1e-5;
        plot(gri_k_rob(vp), log10(EE_EGM2(j,vp)/100), 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('$z_%d$',j));
    end
    title('EGM ($k_{max}=200$)', 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('$\log_{10}$ Error', 'Interpreter', 'latex');
    legend('Location','northeast','Interpreter','latex','FontSize',6); grid on; hold off;
end
sgtitle('Q1: Euler Equation Errors', 'Interpreter', 'latex', 'FontSize', 14);


% --- Figure 4: Combined PE Wealth Distributions (2x2) ---
fig_wealth_pe = figure('Name', 'Q1 Combined: PE Wealth Distributions');
fig_wealth_pe.Position = [50, 50, 1200, 800];

subplot(2,2,1);
histogram(k_vec_VFI, 100, 'Normalization','pdf', 'FaceColor',[0.2 0.4 0.6], 'EdgeColor','none');
title('VFI ($k_{max}=10$)', 'Interpreter', 'latex');
xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex'); grid on;

subplot(2,2,2);
histogram(k_vec_EGM1, 100, 'Normalization','pdf', 'FaceColor',[0.6 0.2 0.2], 'EdgeColor','none');
title('EGM ($k_{max}=10$)', 'Interpreter', 'latex');
xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex'); grid on;

if run_robustness
    if run_VFI_robustness
        subplot(2,2,3);
        histogram(k_vec_VFI2, 100, 'Normalization','pdf', 'FaceColor',[0.4 0.2 0.6], 'EdgeColor','none');
        title('VFI ($k_{max}=200$)', 'Interpreter', 'latex');
        xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex'); grid on;
    end
    subplot(2,2,4);
    histogram(k_vec_EGM2, 100, 'Normalization','pdf', 'FaceColor',[0.2 0.6 0.4], 'EdgeColor','none');
    title('EGM ($k_{max}=200$)', 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex'); grid on;
end
sgtitle('Q1: Stationary Wealth Distributions (PE)', 'Interpreter', 'latex', 'FontSize', 14);


% --- Figure 5: Combined GE Wealth Distributions (1x2) ---
fig_wealth_ge = figure('Name', 'Q2 Combined: GE Wealth Distributions');
fig_wealth_ge.Position = [50, 50, 1200, 400];

subplot(1,2,1); hold on;
histogram(k_sim_GE1, 60, 'Normalization','pdf', 'FaceColor',[0.2 0.4 0.6], 'EdgeColor','w', 'DisplayName','Histogram');
bw1 = std(k_sim_GE1)/3;
[f1, x1] = ksdensity(k_sim_GE1, 'Bandwidth', bw1);
plot(x1, f1, 'r-', 'LineWidth', 2, 'DisplayName', 'KDE');
title('Baseline ($k_{max}=10$)', 'Interpreter', 'latex');
xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex');
legend('Location','northeast'); grid on; hold off;

if run_robustness
    subplot(1,2,2); hold on;
    histogram(k_sim_GE2, 60, 'Normalization','pdf', 'FaceColor',[0.2 0.6 0.4], 'EdgeColor','w', 'DisplayName','Histogram');
    bw2 = std(k_sim_GE2)/3;
    [f2, x2] = ksdensity(k_sim_GE2, 'Bandwidth', bw2);
    plot(x2, f2, 'r-', 'LineWidth', 2, 'DisplayName', 'KDE');
    title('Robust ($k_{max}=200$)', 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex');
    legend('Location','northeast'); grid on; hold off;
end
sgtitle('Q2: GE Stationary Wealth Distributions ($R^*$)', 'Interpreter', 'latex', 'FontSize', 14);



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
if run_robustness
    if run_VFI_robustness
        fprintf('%-30s | %-12.2f | %-16.6f | %-15.4f | %-15.4f\n', '3. Robust VFI (k_max=200)', time_VFI_PE2, max_EE_VFI2, mean_VFI2, gini_VFI2);
    end
    fprintf('%-30s | %-12.2f | %-16.6f | %-15.4f | %-15.4f\n', '4. Robust EGM (k_max=200)', time_EGM_PE2, max_EE_EGM2, mean_EGM2, gini_EGM2);
end
fprintf('===================================================================================================\n');

fprintf('\n===================================================================================================\n');
fprintf('TABLE 2: GENERAL EQUILIBRIUM RESULTS\n');
fprintf('===================================================================================================\n');
fprintf('%-25s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n', 'Setup', 'Equil R*', 'Equil w*', 'Agg K*', 'Gini', 'P90/P10', 'P99/P1');
fprintf('---------------------------------------------------------------------------------------------------\n');
fprintf('%-25s | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', 'Q2 Baseline (k_max=10)', R_star_1, w_star_1, K_star_1, gini_GE1, ratio_90_10_GE1, ratio_99_1_GE1);
if run_robustness
    fprintf('%-25s | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', 'Q2 Robust (k_max=200)', R_star_2, w_star_2, K_star_2, gini_GE2, ratio_90_10_GE2, ratio_99_1_GE2);
end
fprintf('===================================================================================================\n');

% =========================================================================
%% Question 3: Heterogeneous Asset Returns
% =========================================================================
fprintf('\n=================================================================================\n');
fprintf('--- Question 3: Heterogeneous Asset Returns ---\n');
fprintf('=================================================================================\n');

% Define the parameter setups for consistent comparison
if run_robustness
    setups = struct('name', {'Setup 1 (k\_max=10, No Extrap)', 'Setup 2 (k\_max=10, Extrap)', 'Setup 3 (k\_max=200, No Extrap)'}, ...
                    'k_max', {10, 10, 200}, ...
                    'extrapolate', {false, true, false});
else
    setups = struct('name', {'Setup 1 (k\_max=10, No Extrap)'}, ...
                    'k_max', {10}, ...
                    'extrapolate', {false});
end

% Solver config for Heterogeneous Returns
cpar.N_het = 25;
cpar.maxit = 100;
par.alpha = 0.33;
par.w = 1;

% Arrays to store Results for the final table

Q3_results = struct('r_bar', zeros(numel(setups), 2), 'w', zeros(numel(setups), 2), 'K', zeros(numel(setups), 2), 'Gini', zeros(numel(setups), 2), 'p90_10', zeros(numel(setups), 2), 'p99_1', zeros(numel(setups), 2));

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

    % Storage for combining rho_r=0 and rho_r=0.9 side by side
    kpol_het_cases = cell(2,1);
    cpol21_cases = cell(2,1);
    cpol25_cases = cell(2,1);
    k_sim_cases = cell(2,1);
    gri_k_q3 = gri.k;

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
    
    % Save policy data for combined plots
    kpol_het_cases{case_idx} = solving_EGM_het.kpol_k;
    cpol21_cases{case_idx} = (1 + par.r_bar + gri_het.r(21)) * gri.k(:) + par.w * gri_het.z(21) - solving_EGM_het.kpol_k(:, 21);
    cpol25_cases{case_idx} = (1 + par.r_bar + gri_het.r(25)) * gri.k(:) + par.w * gri_het.z(25) - solving_EGM_het.kpol_k(:, 25);

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
    
    % Save wealth simulation for combined plot
    k_sim_cases{case_idx} = k_sim_GE_het;
    
    % Pareto Tail (accumulate on same figure)
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

% --- Combined Q3 Figures for this setup (rho=0 vs rho=0.9) ---

% Combined Savings Policy (1x2)
figure('Name', sprintf('Q3 Combined Savings Policy (%s)', current_setup.name));
for ci = 1:2
    subplot(1,2,ci); hold on;
    plot(gri_k_q3, kpol_het_cases{ci}(:,21), 'r-', 'LineWidth', 2, 'DisplayName', 'High z, Low r');
    plot(gri_k_q3, kpol_het_cases{ci}(:,25), 'b-', 'LineWidth', 2, 'DisplayName', 'High z, High r');
    plot(gri_k_q3, gri_k_q3, 'k--', 'LineWidth', 1.5, 'DisplayName', '45$^\circ$');
    title(sprintf('$\\rho^r = %.1f$', rho_r_cases(ci)), 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('$k''$', 'Interpreter', 'latex');
    legend('Location','northwest','FontSize',7); grid on; hold off;
end
sgtitle(sprintf('Q3 Savings Policy: %s', current_setup.name), 'Interpreter', 'latex', 'FontSize', 14);

% Combined Consumption Policy (1x2)
figure('Name', sprintf('Q3 Combined Consumption Policy (%s)', current_setup.name));
for ci = 1:2
    subplot(1,2,ci); hold on;
    plot(gri_k_q3, cpol21_cases{ci}, 'r-', 'LineWidth', 2, 'DisplayName', 'High z, Low r');
    plot(gri_k_q3, cpol25_cases{ci}, 'b-', 'LineWidth', 2, 'DisplayName', 'High z, High r');
    title(sprintf('$\\rho^r = %.1f$', rho_r_cases(ci)), 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('$c$', 'Interpreter', 'latex');
    legend('Location','northwest','FontSize',7); grid on; hold off;
end
sgtitle(sprintf('Q3 Consumption Policy: %s', current_setup.name), 'Interpreter', 'latex', 'FontSize', 14);

% Combined Wealth Distribution (1x2)
figure('Name', sprintf('Q3 Combined Wealth Distribution (%s)', current_setup.name));
for ci = 1:2
    subplot(1,2,ci); hold on;
    histogram(k_sim_cases{ci}, 60, 'Normalization','pdf', 'FaceColor',[0.3 0.5 0.7], 'EdgeColor','w', 'DisplayName','Histogram');
    bw_q3 = std(k_sim_cases{ci}) / 2;
    [fk, xk] = ksdensity(k_sim_cases{ci}, 'Bandwidth', bw_q3);
    plot(xk, fk, 'r-', 'LineWidth', 2, 'DisplayName', 'KDE');
    title(sprintf('$\\rho^r = %.1f$', rho_r_cases(ci)), 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex');
    legend('Location','northeast'); grid on; hold off;
end
sgtitle(sprintf('Q3 GE Wealth Distribution: %s', current_setup.name), 'Interpreter', 'latex', 'FontSize', 14);

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
if run_robustness
    fprintf('%-30s | %-12.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', 'Q2 Setup 3 (Rep, k_max=200)', R_star_2 - 1, w_star_2, K_star_2, gini_GE2, ratio_90_10_GE2, ratio_99_1_GE2);
end
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

if run_robustness
    fprintf('===================================================================\n');

    %% =====================================================================
    % EXTRA ROBUSTNESS CHECK: Redo Q1, Q2, Q3 with new configuration
    % 1. sigma_eps = 0.05 (instead of sqrt(0.05))
    % 2. kmax = 200, Nkap = 40
    % 3. No extrapolation in simulations (enforces k' <= kmax)
    % =====================================================================

    fprintf('\n\n===================================================================\n');
    fprintf('EXTRA ROBUSTNESS CHECK: New Parameters\n');
    fprintf('===================================================================\n');

    % Set up new parameters
    cpar_ext = cpar;
    cpar_ext.sigmae = 0.05; % Use 0.05 directly (variance vs std dev check)
    cpar_ext.Nkap = 40;
    cpar_ext.max = 200;

    % Re-draw z grid with sigmae instead of sqrt(sigmae)
    [gri_ext.z, gri_ext.prob] = rouwenhorst(cpar_ext.N, cpar_ext.mu, par.rho, cpar_ext.sigmae);

    % Exponentiate and normalize
    gri_ext.z = exp(gri_ext.z);
    [V_ext, D_ext] = eig(gri_ext.prob');
    [~, idx_ext] = min(abs(diag(D_ext) - 1));
    pi_dist_ext = V_ext(:, idx_ext) / sum(V_ext(:, idx_ext));
    mean_z_ext = pi_dist_ext' * gri_ext.z;
    gri_ext.z = gri_ext.z / mean_z_ext;

    % New Capital grid [0, 200] with 40 points
    gri_ext.k = linspace(log(cpar_ext.min + 1), log(cpar_ext.max + 1), cpar_ext.Nkap);
    gri_ext.k = exp(gri_ext.k) - 1;

    par_ext = par;

    %% Q1 Redo (EGM for speed)
    fprintf('\n--- Redoing Q1 (EGM) ---\n');
    par_ext.R = (1/par_ext.beta) * 0.995; % Partial Equilibrium R
    par_ext.w = 1;
    solving_EGM_ext = EGM(par_ext, cpar_ext, gri_ext);

    % Q1 Simulation with Extrapolation
    k_i_ext = zeros(numInd, time);
    k_i_ext(:,1) = 5;

    F_kpol_ext_Q1 = cell(cpar_ext.N, 1);
    for z_state = 1:cpar_ext.N
        F_kpol_ext_Q1{z_state} = griddedInterpolant(gri_ext.k, solving_EGM_ext.kpol_k(:, z_state), 'linear', 'linear');
    end

    for t = 1:time-1
        for z_state = 1:cpar_ext.N
            mask = (z_idx(:, t) == z_state);
            if any(mask)
                % NO EXTRAPOLATION: enforce k' <= kmax
                k_next = F_kpol_ext_Q1{z_state}(k_i_ext(mask, t));
                k_i_ext(mask, t+1) = min(max(0, k_next), cpar_ext.max); % Enforce k' <= kmax
            end
        end
    end
    k_steady_ext = k_i_ext(:, 401:end);
    fprintf('Mean K (Q1 PE): %.4f\n', mean(k_steady_ext(:)));

    % (Extra robustness Q1 policy — see combined figure below)
    kpol_ext_data = solving_EGM_ext.kpol_k;
    k_vec_ext_data = k_steady_ext(:);

    % --- Plot Q1 Wealth Distribution ---
    % (Extra robustness Q1 wealth dist — see combined figure below)


    %% Q2 Redo (GE Bisection)
    fprintf('\n--- Redoing Q2 (GE Bisection) ---\n');
    R_min_ext = 0.995;
    R_max_ext = 1/par_ext.beta;
    diff_ext = tol_GE + 1;
    iter_ext = 0;
    k_sim_ext = 5 * ones(numInd, 1);

    while abs(diff_ext) > tol_GE && iter_ext < max_iter_GE
        iter_ext = iter_ext + 1;
        R_guess = (R_max_ext + R_min_ext)/2;
        
        k_d = (par_ext.alpha / (R_guess - 1 + par_ext.delta))^(1/(1-par_ext.alpha));
        par_ext.R = R_guess;
        par_ext.w = (1 - par_ext.alpha) * k_d^par_ext.alpha;
        
        sol_GE_ext = EGM(par_ext, cpar_ext, gri_ext);
        
        F_kpol_ext_Q2 = cell(cpar_ext.N, 1);
        for z_state = 1:cpar_ext.N
            F_kpol_ext_Q2{z_state} = griddedInterpolant(gri_ext.k, sol_GE_ext.kpol_k(:, z_state), 'linear', 'linear');
        end
        
        k_i_tmp = zeros(numInd, time_GE);
        k_i_tmp(:, 1) = k_sim_ext;
        
        for t = 1:time_GE - 1
            for z_state = 1:cpar_ext.N
                mask = (z_idx_GE(:, t) == z_state);
                if any(mask)
                    % NO EXTRAPOLATION
                    k_next = F_kpol_ext_Q2{z_state}(k_i_tmp(mask, t));
                    k_i_tmp(mask, t+1) = min(max(0, k_next), cpar_ext.max);
                end
            end
        end
        
        k_sim_ext = k_i_tmp(:, end);
        k_supply = mean(k_sim_ext);
        diff_ext = k_supply - k_d;
        
        if diff_ext > 0 
            R_max_ext = R_guess;
        else
            R_min_ext = R_guess;
        end
    end
    fprintf('GE Q2 R* = %.4f | w* = %.4f | K* = %.4f\n', par_ext.R, par_ext.w, k_supply);

    % (Extra robustness Q2 GE wealth dist — see combined figure below)
    k_sim_ext_Q2_data = k_sim_ext;


    %% Q3 Redo (Heterogeneous Returns in GE)
    fprintf('\n--- Redoing Q3 (Heterogeneous Returns in GE) ---\n');
    cpar_ext.Nr = 5;
    cpar_ext.Nz = cpar_ext.N;
    cpar_ext.N_het = cpar_ext.Nz * cpar_ext.Nr;

    Q3_ext_r_bar = zeros(1,2);
    Q3_ext_w = zeros(1,2);
    Q3_ext_K = zeros(1,2);

    rho_r_cases_ext = [0, 0.9];
    k_sim_het_ext_cases = cell(2,1);
    for case_idx = 1:2
        rho_r = rho_r_cases(case_idx);
        sigma_r2 = (1 - rho_r^2) * 0.002;
        
        [r_grid_ext, P_r_ext] = rouwenhorst(cpar_ext.Nr, 0, rho_r, sqrt(sigma_r2));
        
        P_joint_ext = kron(gri_ext.prob, P_r_ext);
        z_joint_ext = kron(gri_ext.z, ones(cpar_ext.Nr, 1));
        r_joint_ext = kron(ones(cpar_ext.Nz, 1), r_grid_ext);
        
        gri_het_ext.k = gri_ext.k;
        gri_het_ext.z = z_joint_ext;
        gri_het_ext.r = r_joint_ext;
        gri_het_ext.prob = P_joint_ext;
        
        r_bar_min_ext = par_ext.delta - 0.02;
        r_bar_max_ext = (0.985 / par_ext.beta) - 1 + 0.05;
        diff_het_ext = 1e-3 + 1;
        iter_het_ext = 0;
        
        k_sim_het_ext = 5 * ones(numInd, 1);
        
        r_idx_ext = zeros(numInd, time_GE);
        r_idx_ext(:, 1) = 3;
        P_cum_r_ext = cumsum(P_r_ext, 2);
        rng(456 + case_idx);
        draws_r_ext = rand(numInd, time_GE);
        
        for t = 1:time_GE - 1
            for r_state = 1:cpar_ext.Nr
                mask = (r_idx_ext(:, t) == r_state);
                if any(mask)
                    r_idx_ext(mask, t+1) = sum(draws_r_ext(mask, t) > P_cum_r_ext(r_state, :), 2) + 1;
                end
            end
        end
        
        het_idx_ext = (z_idx_GE - 1) * cpar_ext.Nr + r_idx_ext;
        
        while abs(diff_het_ext) > 1e-3 && iter_het_ext < 100
            iter_het_ext = iter_het_ext + 1;
            par_ext.r_bar = (r_bar_max_ext + r_bar_min_ext) / 2;
            r_k = par_ext.r_bar + par_ext.delta;
            
            if r_k <= 0
                r_bar_min_ext = par_ext.r_bar; 
                continue; 
            end
            
            k_d = (par_ext.alpha / r_k)^(1 / (1 - par_ext.alpha));
            par_ext.w = (1 - par_ext.alpha) * k_d^par_ext.alpha;
            
            sol_het_ext = EGM_het(par_ext, cpar_ext, gri_het_ext);
            
            F_kpol_ext_Q3 = cell(cpar_ext.N_het, 1);
            for s = 1:cpar_ext.N_het
                F_kpol_ext_Q3{s} = griddedInterpolant(gri_ext.k, sol_het_ext.kpol_k(:, s), 'linear', 'linear');
            end
            
            k_i_het_tmp = zeros(numInd, time_GE);
            k_i_het_tmp(:, 1) = k_sim_het_ext;
            
            for t = 1:time_GE - 1
                for s = 1:cpar_ext.N_het
                    mask = (het_idx_ext(:, t) == s);
                    if any(mask)
                        % NO EXTRAPOLATION
                        k_next = F_kpol_ext_Q3{s}(k_i_het_tmp(mask, t));
                        k_i_het_tmp(mask, t+1) = min(max(0, k_next), cpar_ext.max);
                    end
                end
            end
            
            k_sim_het_ext = k_i_het_tmp(:, end);
            k_supply = mean(k_sim_het_ext);
            diff_het_ext = k_supply - k_d;
            
            if diff_het_ext > 0
                r_bar_max_ext = par_ext.r_bar;
            else
                r_bar_min_ext = par_ext.r_bar;
            end
        end
        
        Q3_ext_r_bar(case_idx) = par_ext.r_bar;
        Q3_ext_w(case_idx) = par_ext.w;
        Q3_ext_K(case_idx) = k_supply;
        fprintf('  GE Q3 Case %d (rho_r=%.1f): r_bar* = %.4f, K* = %.4f\n', case_idx, rho_r, par_ext.r_bar, k_supply);

        % Save Q3 ext wealth simulation per case
        k_sim_het_ext_cases{case_idx} = k_sim_het_ext;
    end

    fprintf('\n===================================================================\n');
    fprintf('TABLE: EXTRA Robustness Check — General Equilibrium Results\n');
    fprintf('===================================================================\n');
    fprintf('%-20s | %-10s | %-10s | %-10s\n', 'Case', 'r_bar*', 'w*', 'K*');
    fprintf('-------------------------------------------------------------------\n');
    for case_idx = 1:2
        fprintf('%-20s | %-10.4f | %-10.4f | %-10.4f\n', ...
            sprintf('rho_r = %.1f', rho_r_cases(case_idx)), ...
            Q3_ext_r_bar(case_idx), Q3_ext_w(case_idx), Q3_ext_K(case_idx));
    end
    fprintf('===================================================================\n');

    % --- Combined Extra Robustness Figure (2x2) ---
    figure('Name', 'Extra Robustness Combined');
    set(gcf, 'Position', [50, 50, 1200, 800]);
    colors_ext = lines(cpar_ext.N);

    % (1,1) Q1 PE Policy Function
    subplot(2,2,1); hold on;
    for j = 1:cpar_ext.N
        plot(gri_ext.k, kpol_ext_data(:,j), 'LineWidth', 1.5, 'Color', colors_ext(j,:));
    end
    plot(gri_ext.k, gri_ext.k, 'k--', 'LineWidth', 1.5);
    title('Q1 PE Policy $k''(k,z)$', 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('$k''$', 'Interpreter', 'latex');
    grid on; hold off;

    % (1,2) Q1 PE Wealth Distribution
    subplot(2,2,2);
    histogram(k_vec_ext_data, 100, 'Normalization','pdf', 'FaceColor',[0.4 0.2 0.6], 'EdgeColor','none');
    title('Q1 PE Wealth Distribution', 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex');
    grid on;

    % (2,1) Q2 GE Wealth Distribution
    subplot(2,2,3); hold on;
    histogram(k_sim_ext_Q2_data, 60, 'Normalization','pdf', 'FaceColor',[0.6 0.2 0.4], 'EdgeColor','w', 'DisplayName','Histogram');
    bw_ext2 = std(k_sim_ext_Q2_data)/3;
    [fe2, xe2] = ksdensity(k_sim_ext_Q2_data, 'Bandwidth', bw_ext2);
    plot(xe2, fe2, 'r-', 'LineWidth', 2, 'DisplayName', 'KDE');
    title('Q2 GE Wealth Distribution', 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex');
    legend('Location','northeast'); grid on; hold off;

    % (2,2) Q3 GE Wealth Distribution (both rho cases)
    subplot(2,2,4); hold on;
    for ci = 1:2
        bw_q3e = std(k_sim_het_ext_cases{ci})/3;
        [fq3, xq3] = ksdensity(k_sim_het_ext_cases{ci}, 'Bandwidth', bw_q3e);
        plot(xq3, fq3, 'LineWidth', 2, 'DisplayName', sprintf('$\rho^r = %.1f$', rho_r_cases_ext(ci)));
    end
    title('Q3 GE Wealth Distribution', 'Interpreter', 'latex');
    xlabel('$k$', 'Interpreter', 'latex'); ylabel('Density', 'Interpreter', 'latex');
    legend('Location','northeast','Interpreter','latex'); grid on; hold off;

    sgtitle('Extra Robustness ($k_{max}=200, N_k=40, \sigma_e=0.05$, No Extrap)', 'Interpreter', 'latex', 'FontSize', 14);
end