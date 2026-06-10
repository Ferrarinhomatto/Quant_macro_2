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


%% a) Parameters
beta  = 0.95^(1/12);
gamma = 2;
r     = 1 + (0.03/12);
w     = 0.98;
A     = 1;
z     = 0.3;
sigma = 0.03;
u_ss  = 0.1;
f     = (sigma/u_ss) - sigma;
theta = 1;
alpha = 0.5;

% Convergence/grid parameters
Nb     = 100;
bmax  = 5;
bmin  = -3;
% _j = {1,2} = {e,u}

%% b) Household problem

% Grid and Income
bgrid = linspace(bmin, bmax, Nb)';
ygrid = [w, z]; 

% Transition Matrix Pi(j, jp) = Prob(x'=jp | x=j)
% State 1: e, State 2: u
Pi = [1-sigma, sigma; ...
      f,       1-f];

% Solve Household Problem (Policy Function)

% Solving for b'=h(b,x)

% Initial guess

% From budget constraint and c = y(x) we get b' = rb

bp_old = r*[bgrid, bgrid];

% While loop for convergence:

tol = 1e-6;
dist = 10;
maxiter = 10000;
iter = 0;

while dist > tol && iter < maxiter

    iter = iter + 1;
    [bp_new, c] = egmUpdateSS(bp_old, bgrid, ygrid, Pi, beta, r, gamma, bmin);
    dist = max(abs(bp_new(:) - bp_old(:)));
    bp_old = bp_new;

    % Visual feedback is helpful for debugging
end

fprintf('EGM converged in %d iterations with a final distance of %e.\n', iter, dist);

bp_pol = bp_new;
c_pol = c;

% Helper function to plot without clutter in main script:
plot_policy_functions(bgrid, bp_pol, c_pol);

%% b.1) Solving again with b_max = 10 instead of b_max=5

bmax = 10;

% Grid and Income
bgrid_2 = linspace(bmin, bmax, Nb)';

% Solve Household Problem (Policy Function)

% Solving for b'=h(b,x)

% Initial guess

% From budget constraint and c = y(x) we get b' = rb

bp_old_2 = r*[bgrid_2, bgrid_2];

% While loop for convergence:

tol = 1e-6;
dist = 10;
maxiter = 10000;
iter = 0;

while dist > tol && iter < maxiter

    iter = iter + 1;
    [bp_new_2, c_2] = egmUpdateSS(bp_old_2, bgrid_2, ygrid, Pi, beta, r, gamma, bmin);
    dist = max(abs(bp_new_2(:) - bp_old_2(:)));
    bp_old_2 = bp_new_2;

    % Visual feedback is helpful for debugging
end

fprintf('EGM converged in %d iterations with a final distance of %e.\n', iter, dist);

bp_pol_2 = bp_new_2;
c_pol_2 = c_2;

% Helper function to plot without clutter in main script:
plot_policy_functions2(bgrid_2, bp_pol_2, c_pol_2);

%% b.2) Household problem for Final Baseline (bmax = 24)

bmax_3 = 24;
bgrid_3 = linspace(bmin, bmax_3, Nb)';
bp_old_3 = r * [bgrid_3, bgrid_3];

tol = 1e-6;
dist_val = 10;
maxiter = 10000;
iter_3 = 0;

while dist_val > tol && iter_3 < maxiter
    iter_3 = iter_3 + 1;
    [bp_new_3, c_3] = egmUpdateSS(bp_old_3, bgrid_3, ygrid, Pi, beta, r, gamma, bmin);
    dist_val = max(abs(bp_new_3(:) - bp_old_3(:)));
    bp_old_3 = bp_new_3;
end
fprintf('EGM (bmax=24) converged in %d iterations.\n', iter_3);
bp_pol_3 = bp_new_3;
c_pol_3 = c_3;

%% c) Stationary Distribution

% Using provided Young (2010) method helper function
g_ss = solveErgDist(bp_pol, bgrid, f, sigma);

% Stationary distribution stats

% 1. Unemployment Rate
% Left column: employed ... Right column: unemployed
u_ergodic = sum(g_ss(:, 2)); 
fprintf('Unemployment Rate (Implied): %.4f\n', u_ergodic);
fprintf('Unemployment Rate (Formula): %.4f\n', u_ss);

% 2. Mean Assets (Overall and Conditional)
% Overall mean = sum(b * g(b,x)) across all b and x
mean_b_overall = sum(bgrid .* g_ss, 'all'); 

% Conditional mean = sum(b * g(b,x)) / sum(g(b,x)) for a specific x
mean_b_e = sum(bgrid .* g_ss(:, 1)) / sum(g_ss(:, 1));
mean_b_u = sum(bgrid .* g_ss(:, 2)) / sum(g_ss(:, 2));

fprintf('\nMean Assets (Overall):    %.4f\n', mean_b_overall);
fprintf('Mean Assets (Employed):   %.4f\n', mean_b_e);
fprintf('Mean Assets (Unemployed): %.4f\n', mean_b_u);

% 3. Share of households at the borrowing constraint (b_min index 1)
% Mass at b_min divided by the total mass of the state
share_bc_e = g_ss(1, 1) / sum(g_ss(:, 1));
share_bc_u = g_ss(1, 2) / sum(g_ss(:, 2));

fprintf('\nShare at Borrowing Constraint (Employed):   %.2f%%\n', share_bc_e * 100);
fprintf('Share at Borrowing Constraint (Unemployed): %.2f%%\n', share_bc_u * 100);
fprintf('======================================================\n');

% 4. Plot the distribution using a new helper function for cleanliness
plot_distribution(bgrid, g_ss);

%% c.1) Stationary Distribution for bmax=10

% Using provided Young (2010) method helper function with the new policy and grid
g_ss_2 = solveErgDist(bp_pol_2, bgrid_2, f, sigma);

% 1. Unemployment Rate
u_ergodic_2 = sum(g_ss_2(:, 2)); 
fprintf('Unemployment Rate (Implied): %.4f\n', u_ergodic_2);

% 2. Mean Assets (Overall and Conditional)
% Overall mean = sum(b * g(b,x)) across all b and x
mean_b_overall_2 = sum(bgrid_2 .* g_ss_2, 'all'); 

% Conditional mean = sum(b * g(b,x)) / sum(g(b,x)) for a specific x
mean_b_e_2 = sum(bgrid_2 .* g_ss_2(:, 1)) / sum(g_ss_2(:, 1));
mean_b_u_2 = sum(bgrid_2 .* g_ss_2(:, 2)) / sum(g_ss_2(:, 2));

fprintf('\nMean Assets (Overall):    %.4f\n', mean_b_overall_2);
fprintf('Mean Assets (Employed):   %.4f\n', mean_b_e_2);
fprintf('Mean Assets (Unemployed): %.4f\n', mean_b_u_2);

% 3. Share of households at the borrowing constraint (b_min index 1)
% Mass at b_min divided by the total mass of the state
share_bc_e_2 = g_ss_2(1, 1) / sum(g_ss_2(:, 1));
share_bc_u_2 = g_ss_2(1, 2) / sum(g_ss_2(:, 2));

fprintf('\nShare at Borrowing Constraint (Employed):   %.2f%%\n', share_bc_e_2 * 100);
fprintf('Share at Borrowing Constraint (Unemployed): %.2f%%\n', share_bc_u_2 * 100);
fprintf('======================================================\n');

% 4. Plot the new distribution using a new helper function
plot_distribution2(bgrid_2, g_ss_2);

%% c.2) Stationary Distribution for Final Baseline (bmax=24)

g_ss_3 = solveErgDist(bp_pol_3, bgrid_3, f, sigma);

mean_b_overall_3 = sum(bgrid_3 .* g_ss_3, 'all'); 
mean_b_e_3 = sum(bgrid_3 .* g_ss_3(:, 1)) / sum(g_ss_3(:, 1));
mean_b_u_3 = sum(bgrid_3 .* g_ss_3(:, 2)) / sum(g_ss_3(:, 2));

share_bc_e_3 = g_ss_3(1, 1) / sum(g_ss_3(:, 1));
share_bc_u_3 = g_ss_3(1, 2) / sum(g_ss_3(:, 2));

fprintf('\n======================================================\n');
fprintf('--- Stats for Final Baseline (bmax=24) ---\n');
fprintf('======================================================\n');
fprintf('Mean Assets (Overall):    %.4f\n', mean_b_overall_3);
fprintf('Mean Assets (Employed):   %.4f\n', mean_b_e_3);
fprintf('Mean Assets (Unemployed): %.4f\n', mean_b_u_3);
fprintf('Share at Borrowing Constraint (Employed):   %.4f%%\n', share_bc_e_3 * 100);
fprintf('Share at Borrowing Constraint (Unemployed): %.4f%%\n', share_bc_u_3 * 100);
fprintf('======================================================\n');


%% 1.d) Recover labour market objects and calibrate kappa

% 1. Analytically solve for implied match efficiency m
% From f = m * theta^(1-alpha), we get m = f / (theta^(1-alpha))
m = f / (theta^(1 - alpha));

% 2. Compute vacancy filling rate q = m * theta^(-alpha)
q = m * (theta^(-alpha));

% 3. Compute job value J
% Formula: J = (A - w) / (1 - beta_f * (1 - sigma))
% Note: In our parameters, beta_f is just beta.
J = (A - w) / (1 - beta * (1 - sigma));

% 4. Calibrate the vacancy cost kappa from free entry
% Formula: kappa = beta_f * q * J
kappa = beta * q * J;

% Print the results
fprintf('\n======================================================\n');
fprintf('--- Question 1d: Labour Market Calibration ---\n');
fprintf('======================================================\n');
fprintf('Match Efficiency (m):   %.4f\n', m);
fprintf('Vacancy Filling Rate (q): %.4f\n', q);
fprintf('Job Value (J):          %.4f\n', J);
fprintf('Vacancy Cost (kappa):   %.4f\n', kappa);
fprintf('======================================================\n');

%% 1.e) Closing the asset market

% Aggregate bond demand is the overall mean assets from the true (bmax=24) distribution
Bd = mean_b_overall_3; 
Bs = Bd;

fprintf('\n======================================================\n');
fprintf('--- Question 1e: Asset Market Equilibrium ---\n');
fprintf('======================================================\n');
fprintf('Equilibrium Bond Supply (Bs): %.4f\n', Bs);
fprintf('======================================================\n');

clear restoreDir; 
save('ws_steady.mat');

