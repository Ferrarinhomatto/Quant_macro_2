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
cpar.sigmae = 0.005;

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


%% Part 3 - Simulating 2000 individuals over time
numInd = 2000;
time = 1400;

% Matrices
i_i = zeros(numInd, time); % Assuming you use this elsewhere
z_i = zeros(numInd, time);
c_i = zeros(numInd, time);
k_i = zeros(numInd, time);
c_i_int = zeros(numInd, cpar.Nkap);
k_i_int = zeros(numInd, cpar.Nkap);

% Initial Conditions
z_i(:,1) = gri.z(3);
k_i(:,1) = 5;

for t = 1:time
    for j = 1:numInd
        
        % 1. Interpolate policy functions for the current z_i
        for k = 1:cpar.Nkap
              c_i_int(j,k) = interp_kp(gri.z, cpol(:,k), z_i(j, t));
              k_i_int(j,k) = interp_kp(gri.z, kpol(:,k), z_i(j,t));
        end
        
        % 2. Interpolate optimal k and c for the current k_i
        c_i(j,t) = interp_kp(gri.k, c_i_int(j, :), k_i(j,t));
         
        % 3. Calculate next period states ONLY if we are not in the last period
        if t < time
             k_i(j, t+1) = interp_kp(gri.k, k_i_int(j, :), k_i(j,t));
             
             % FIX: Use randn() to get a scalar, not randn(j)!
             z_i(j, t+1) = exp(par.rho * log(z_i(j,t)) + cpar.sigmae * randn());
        end
        
    end
end

%% Plotting the Wealth Distribution and Calculating Statistics

% 1. Discard the first 400 periods (burn-in) 
drop = 400;
k_steady = k_i(:, drop+1:end);

% Flatten the matrix into a single vector of stationary asset holdings
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

%% Grid for x

x_min = min(gri.z)  + par.R * gri.k(1);

x_max = max(gri.z) + par.R * gri.k(end);

gri.x = linspace(x_min, x_max, 100);


%% 1.B - EMG 

% We have a function for it


solving_EGM = EGM(par, cpar, gri);


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

