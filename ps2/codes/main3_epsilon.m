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

%% 3.a Monetary Policy Response
% Find epsilon that stabilizes economy at b_min = -2.5

css_data = load('ws_steady.mat');

w_css = css_data.w;
r_css = css_data.r;
beta  = css_data.beta;
sigma = css_data.sigma;
f_css = css_data.f;
z     = css_data.z;
gamma = css_data.gamma;
Nb    = css_data.Nb;
bmax  = css_data.bmax_3;  % bmax = 24
Bs    = css_data.Bs;      % Steady state bond supply

% New constraint
bmin_new = -2.5;
bgrid_new = linspace(bmin_new, bmax, Nb)';

%% Find natural interest rate (r*)
% Labor market fixed at SS (w_css, f_css)

ygrid_fixed = [w_css, z];
Pi_fixed = [1-sigma, sigma; f_css, 1-f_css];

r_star = r_css;
tol_r = 1e-5;
err = 10;
lambda = 0.001;
max_iter = 500;
iter = 0;

disp('Finding r*...');

while abs(err) > tol_r && iter < max_iter
    iter = iter + 1;
    
    bp_old = r_star * [bgrid_new, bgrid_new];
    dist_val = 10; iterhh = 0;
    while dist_val > 1e-6 && iterhh < 2000
        iterhh = iterhh + 1;
        [bp_new, ~] = egmUpdateSS(bp_old, bgrid_new, ygrid_fixed, Pi_fixed, beta, r_star, gamma, bmin_new);
        dist_val = max(abs(bp_new(:) - bp_old(:)));
        bp_old = bp_new;
    end
    
    g_ss_star = solveErgDist(bp_new, bgrid_new, f_css, sigma);
    Bd_star = sum(bgrid_new .* g_ss_star, 'all');
    
    err = Bs - Bd_star;
    r_star = r_star + lambda * err;
end

fprintf('r^*: %.6f\n', r_star);

%% Required Taylor rule shock (\epsilon)

epsilon_opt = r_star / r_css;

fprintf('epsilon: %.6f\n', epsilon_opt);

%% GE loop to verify stabilization

A      = css_data.A;
alpha  = css_data.alpha;
m      = css_data.m;
kappa  = css_data.kappa;
xi     = 1.2;
phi_pi = 1.5;
pi_ss  = 1 + (0.02/12);

% Reset GE loop guess
r = r_css;
err_ge = 10;
lambda_ge = 0.001;
iter_ge = 0;

disp('Checking GE under epsilon_opt...');

while abs(err_ge) > tol_r && iter_ge < max_iter
    iter_ge = iter_ge + 1;
    
    % Monetary variables with optimally accommodated epsilon
    pi_ge = pi_ss * (r / (r_css * epsilon_opt))^(1 / (phi_pi - 1));
    w_ge  = w_css * (pi_ss/pi_ge)^xi;
    
    % Labor market variables
    J_ge = (A - w_ge) / (1 - beta * (1 - sigma));
    theta_ge = (kappa / (beta * m * J_ge))^(-1/alpha);
    f_ge = m * theta_ge^(1-alpha);
    f_ge = min(max(f_ge, 0.001), 0.99); % safeguard
    
    % Household problem
    ygrid_ge = [w_ge, z];
    Pi_ge = [1-sigma, sigma; f_ge, 1-f_ge];
    bp_old_ge = r * [bgrid_new, bgrid_new]; 
    
    dist_hh = 10; iterhh_ge = 0;
    while dist_hh > 1e-6 && iterhh_ge < 2000
        iterhh_ge = iterhh_ge + 1;
        [bp_new_ge, ~] = egmUpdateSS(bp_old_ge, bgrid_new, ygrid_ge, Pi_ge, beta, r, gamma, bmin_new);
        dist_hh = max(abs(bp_new_ge(:) - bp_old_ge(:)));
        bp_old_ge = bp_new_ge;
    end
    
    % Stationary distribution
    g_ss_ge = solveErgDist(bp_new_ge, bgrid_new, f_ge, sigma);
    Bd_ge = sum(bgrid_new .* g_ss_ge, 'all');
    
    err_ge = Bs - Bd_ge;
    r = r + lambda_ge * err_ge;
end

i_ge = r * pi_ge;
u_ge = sigma / (sigma + f_ge);

% Checks
fprintf('u: %.2f%%\n', u_ge * 100);

%% ================================================================
%  FIGURES FOR Q3.1 -- Monetary Policy Stabilisation
%  ================================================================

bgrid_css = css_data.bgrid_3;
g_ss_css  = css_data.g_ss_3;

col_css = [0.20 0.45 0.70];
col_new = [0.85 0.33 0.10];

%% Figure 1: Stationary Distribution Overlay
figure('Name','Q3.1 -- Distribution Overlay','NumberTitle','off');
set(gcf,'Position',[80 80 960 440]);

subplot(1,2,1); hold on;
area(bgrid_css, g_ss_css(:,1),'FaceColor',col_css,'FaceAlpha',0.50,'EdgeColor',col_css,'LineWidth',1.5,'DisplayName','Calibrated SS');
area(bgrid_new, g_ss_ge(:,1), 'FaceColor',col_new,'FaceAlpha',0.40,'EdgeColor',col_new,'LineWidth',1.5,'DisplayName','Monetary Stab.\ ($\epsilon^*$)');
xline(-3,   ':','Color',[0.5 0.5 0.5],'LineWidth',1.1,'HandleVisibility','off');
xline(-2.5,'--k','LineWidth',1.2,'HandleVisibility','off');
text(-2.4, max(g_ss_css(:,1))*0.55,'$b_{min}''$','Interpreter','latex','FontSize',9);
xlabel('Bond holdings $b$'); ylabel('Prob.\ mass $g(b,e)$');
title('Employed Households ($x=e$)'); legend('Location','best','FontSize',8);
xlim([-3.5 8]); grid on; box on;

subplot(1,2,2); hold on;
area(bgrid_css, g_ss_css(:,2),'FaceColor',col_css,'FaceAlpha',0.50,'EdgeColor',col_css,'LineWidth',1.5,'DisplayName','Calibrated SS');
area(bgrid_new, g_ss_ge(:,2), 'FaceColor',col_new,'FaceAlpha',0.40,'EdgeColor',col_new,'LineWidth',1.5,'DisplayName','Monetary Stab.\ ($\epsilon^*$)');
xline(-3,   ':','Color',[0.5 0.5 0.5],'LineWidth',1.1,'HandleVisibility','off');
xline(-2.5,'--k','LineWidth',1.2,'HandleVisibility','off');
text(-2.4, max(g_ss_css(:,2))*0.55,'$b_{min}''$','Interpreter','latex','FontSize',9);
xlabel('Bond holdings $b$'); ylabel('Prob.\ mass $g(b,u)$');
title('Unemployed Households ($x=u$)'); legend('Location','best','FontSize',8);
xlim([-3.5 8]); grid on; box on;

sgtitle('Q3.1 -- Stationary Distribution: SS vs.\ Monetary Stabilisation ($\epsilon^*$)','FontWeight','bold');

%% Figure 2: Bond Market Equilibrium  (sweep r, hold labour at SS)
fprintf('\n  Sweeping r to build bond demand curve (Q3.1 bond market)...\n');

r_sweep  = linspace(1.0010, 1.0040, 30);
Bd_sweep = zeros(1,30);

for k = 1:30
    r_k   = r_sweep(k);
    bp_sw = r_k * [bgrid_new, bgrid_new]; % init once per r_k
    d_sw  = 10;  it_sw = 0;
    while d_sw > 1e-6 && it_sw < 2500
        it_sw = it_sw + 1;
        [bp_sw_new, ~] = egmUpdateSS(bp_sw, bgrid_new, ygrid_fixed, Pi_fixed, beta, r_k, gamma, bmin_new);
        d_sw  = max(abs(bp_sw_new(:) - bp_sw(:)));
        bp_sw = bp_sw_new;
    end
    g_sw        = solveErgDist(bp_sw, bgrid_new, f_css, sigma);
    Bd_sweep(k) = sum(bgrid_new .* g_sw, 'all');
end

figure('Name','Q3.1 -- Bond Market','NumberTitle','off');
set(gcf,'Position',[200 80 660 500]);
hold on;

% Bond demand curve  (upward-sloping: higher r -> more savings)
plot(Bd_sweep, r_sweep, 'b-', 'LineWidth', 2.5, 'DisplayName', '$B^d(r)$, new $b_{min}$');

% Fixed supply (vertical line)
plot([Bs Bs], [r_sweep(1) r_sweep(end)], '-', 'Color',[0.13 0.55 0.13], 'LineWidth', 2.5, 'DisplayName', '$B^s$ (fixed supply)');

% Reference rates
yline(r_css,  '--','Color',[0.5 0.5 0.5],'LineWidth',1.4,'HandleVisibility','off');
yline(r_star, '-.','Color',col_new,'LineWidth',1.4,'HandleVisibility','off');
text(min(Bd_sweep)+0.01, r_css+0.00008,'$r_{ss}$','Interpreter','latex','FontSize',11,'Color',[0.5 0.5 0.5]);
text(min(Bd_sweep)+0.01, r_star-0.00015,'$r^*$','Interpreter','latex','FontSize',11,'Color',col_new);

% Equilibrium marker
scatter(Bs, r_star, 110, 'r', 'filled', 'MarkerEdgeColor','k', 'DisplayName','Equilibrium $(B^s,\,r^*)$');

xlabel('Bond Quantity $B$'); ylabel('Real Interest Rate $r$');
title('Q3.1 -- Bond Market Equilibrium under Monetary Stabilisation');
legend('Location','southeast','FontSize',9);
grid on; box on;
