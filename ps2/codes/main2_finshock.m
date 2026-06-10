%% Problem set 1 - QM2 - Ferrari Jimenez

% ---------------------------------------

% House cleaning

clear; clc; close all;

rng(1);

% Easy directory handling for reviewer

oldDir = pwd;
restoreDir = onCleanup(@() cd(oldDir)); 

projectRoot = fileparts(mfilename('fullpath'));
if ~isempty(projectRoot)
    cd(projectRoot);
end

% Ensure the figures directory exists relative to the codes folder
fig_dir = '../figures/';
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

% Set LaTeX interpreters (for plots)
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');

% add the path of the functions

addpath("functions/");

% -----------------------------------------

%% a) Housekeeping and parameters

% Loading steady state results
css_data = load('ws_steady.mat');

% Explicitly loading needed variables from calibrated steady state (CSS)
w_css = css_data.w;
r_css = css_data.r;
A     = css_data.A;
beta  = css_data.beta;
sigma = css_data.sigma;
alpha = css_data.alpha;
m     = css_data.m;
kappa = css_data.kappa;
z     = css_data.z;
gamma = css_data.gamma;
Nb    = css_data.Nb;
bmax  = css_data.bmax_3;  % bmax = 24
Bs    = css_data.Bs;      % Steady state bond supply to target

% New nominal parameters
epsilon = 1;              
xi      = 1.2;
phi_pi  = 1.5;
pi_ss   = 1 + (0.02/12);

% Tightened borrowing constraint
bmin_new = -2.5;

% Initial guess for r from CSS
r = r_css;

% Loop parameters
tol_r    = 1e-5;          % Tolerance for market clearing
err      = 10;            % Initial error
lambda   = 0.001;         % Adjustment parameter
max_iter = 1000;
iter     = 0;

% New grid for b'
bgrid_css = css_data.bgrid_3;

bgrid_new = linspace(bmin_new, bmax, Nb)';

%% The General Equilibrium Loop: Steps (b) to (f)
while abs(err) > tol_r && iter < max_iter
    iter = iter + 1;
    
    %% b) Updated monetary variables and wages
    % Given a guess for r:
    pi = pi_ss * (r / (r_css * epsilon))^(1 / (phi_pi - 1));
    i  = r * pi;
    w  = w_css * (pi_ss/pi)^xi;
    
    %% c) Updated labour market equilibrium
    J = (A-w)/(1-beta*(1-sigma));
    theta = (kappa/(beta*m*J))^(-1/alpha);
    q = m*theta^(-alpha);
    f = m*theta^(1-alpha);
    
    %% d) Solve the household problem

    % Crucially, have to update new w and f (in ygrid and Pi)

    ygrid = [w, z];
    Pi = [1-sigma, sigma; f, 1-f];
    
    bp_old = r * [bgrid_new, bgrid_new]; 

    tol = 1e-6;
    dist_val = 10;
    maxiter = 10000;
    iterhh = 0;
    
    while dist_val > tol && iterhh < maxiter
        iterhh = iterhh + 1;
        [bp_new, c] = egmUpdateSS(bp_old, bgrid_new, ygrid, Pi, beta, r, gamma, bmin_new);
        dist_val = max(abs(bp_new(:) - bp_old(:)));
        bp_old = bp_new;
    end
    
    bp_pol = bp_new;
    c_pol = c;
        
    %% e) Compute the stationary distribution
    % Use the newly computed policy function, grid, and job-finding rate
    g_ss_new = solveErgDist(bp_pol, bgrid_new, f, sigma);
    
    %% f) Update the real interest rate
    % Compute aggregate bond demand using the new distribution
    Bd = sum(bgrid_new .* g_ss_new, 'all');
    
    % Compute market clearing error
    err = Bs - Bd;
    
    % Print progress so you can watch the loop converge
    fprintf('GE Iteration: %d | r: %.6f | err: %.6f\n', iter, r, err);
    
    % Update the real interest rate guess
    r = r + lambda * err;
    
end % End of the GE while loop

fprintf('\n=== NEW STEADY STATE FOUND ===\n');
fprintf('New Real Interest Rate (r): %.6f\n', r);
fprintf('New Unemployment Rate:      %.4f\n', sigma / (sigma + f));

fprintf('\n=== DELIVERABLES REPORT ===\n');
fprintf('Inflation (pi): %.6f (vs %.6f)\n', pi, pi_ss);
fprintf('Nominal Rate (i): %.6f (vs %.6f)\n', i, r_css * pi_ss);
fprintf('Real Wage (w): %.6f (vs %.6f)\n', w, w_css);
fprintf('Market Tightness (theta): %.6f (vs 1.0000)\n', theta);
fprintf('Job Finding Rate (f): %.6f\n', f);

% --- Add to the very end of main2_finshock.m ---
mean_b_overall = sum(bgrid_new .* g_ss_new, 'all'); 
mean_b_e = sum(bgrid_new .* g_ss_new(:, 1)) / sum(g_ss_new(:, 1));
mean_b_u = sum(bgrid_new .* g_ss_new(:, 2)) / sum(g_ss_new(:, 2));

share_bc_e = g_ss_new(1, 1) / sum(g_ss_new(:, 1));
share_bc_u = g_ss_new(1, 2) / sum(g_ss_new(:, 2));

fprintf('\n=== DISTRIBUTIONAL CHANGES ===\n');
fprintf('Mean Assets (Overall):    %.4f\n', mean_b_overall);
fprintf('Mean Assets (Employed):   %.4f\n', mean_b_e);
fprintf('Mean Assets (Unemployed): %.4f\n', mean_b_u);
fprintf('Share at BC (Employed):   %.4f%%\n', share_bc_e * 100);
fprintf('Share at BC (Unemployed): %.4f%%\n', share_bc_u * 100);

% Plotting Section
% Extract original distribution from the loaded workspace
g_ss_css = css_data.g_ss_3;

% Generate only the comparative overlay plot
plot_distribution_overlay(bgrid_css, g_ss_css, bgrid_new, g_ss_new, 'Q2_dist_overlay');

%% g) Parameter Sensitivity Tests (Section 2.d)
test_cases = {
    1.5, 0.0, 'Flexible Wages ($\xi=0$)';
    1.5, 1.5, 'High Wage Rigidity ($\xi=1.5$)';
    2.0, 1.2, 'Aggressive CB ($\phi_\pi=2$)';
    1.2, 1.2, 'Passive CB ($\phi_\pi=1.2$)'
};

fprintf('\n=== PARAMETER SENSITIVITY TESTS (SECTION 2.d) ===\n');
fprintf('%-28s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-4s\n', ...
    'Scenario', 'r', 'pi', 'i', 'w', 'theta', 'f', 'u', 'Iter');
fprintf(repmat('-', 1, 105));
fprintf('\n');

for k = 1:size(test_cases, 1)
    phi_test = test_cases{k, 1};
    xi_test  = test_cases{k, 2};
    name     = test_cases{k, 3};
    
    r = r_css; err = 10; err_old = 10; iter_test = 0;
    tol_ge = 5e-5; tol_hh = 1e-6; max_ge_iter = 500;
    
    if phi_test < 1.3
        lambda_test = 0.0001; 
    else
        lambda_test = 0.002;
    end
    
    while abs(err) > tol_ge && iter_test < max_ge_iter
        iter_test = iter_test + 1;
        
        pi_test = pi_ss * (r / (r_css * epsilon))^(1 / (phi_test - 1));
        w_test  = w_css * (pi_ss/pi_test)^xi_test;
        
        if w_test >= A
            w_test = A - 0.0001; 
        end
        
        J_test = (A - w_test) / (1 - beta*(1 - sigma));
        theta_test = (kappa / (beta * m * J_test))^(-1/alpha);
        f_test = m * theta_test^(1-alpha);
        f_test = min(max(f_test, 0.001), 0.99);
        
        ygrid_test = [w_test, z];
        Pi_test = [1-sigma, sigma; f_test, 1-f_test];
        bp_old_test = r * [bgrid_new, bgrid_new]; 
        
        dist_hh = 10; iter_hh = 0;
        while dist_hh > tol_hh && iter_hh < 2000
            iter_hh = iter_hh + 1;
            [bp_new_test, ~] = egmUpdateSS(bp_old_test, bgrid_new, ygrid_test, ...
                                           Pi_test, beta, r, gamma, bmin_new);
            dist_hh = max(abs(bp_new_test(:) - bp_old_test(:)));
            bp_old_test = bp_new_test;
        end
        
        g_ss_test = solveErgDist(bp_new_test, bgrid_new, f_test, sigma);
        Bd_test = sum(bgrid_new .* g_ss_test, 'all');
        err = Bs - Bd_test;
        
        if iter_test > 1 && sign(err) ~= sign(err_old)
            lambda_test = lambda_test * 0.5;
        end
        err_old = err;
        
        r = r + lambda_test * err;
    end
    
    i_test = r * pi_test;
    u_final = sigma / (sigma + f_test);
    
    % Print the expanded result row
    fprintf('%-28s | %8.6f | %8.6f | %8.6f | %8.6f | %8.6f | %8.6f | %5.2f%% | %d\n', ...
        name, r, pi_test, i_test, w_test, theta_test, f_test, u_final * 100, iter_test);
        
    % Plot and save the distribution for this scenario
    filename = sprintf('Q2_sens_case%d', k);
    plot_distribution_simple(bgrid_new, g_ss_test, filename, name);
end
fprintf('\n');