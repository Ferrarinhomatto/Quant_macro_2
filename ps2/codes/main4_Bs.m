%% Problem set 2 - QM2 - Ferrari Jimenez

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

%% 3.b Fiscal Policy Response
css_data = load('ws_steady.mat');

w_css = css_data.w;
r_css = css_data.r;
beta  = css_data.beta;
sigma = css_data.sigma;
f_css = css_data.f;
z     = css_data.z;
gamma = css_data.gamma;
Nb    = css_data.Nb;

% Retrieve bmax from steady-state (assuming it's named bmax_3 or bmax)
if isfield(css_data, 'bmax_3')
    bmax = css_data.bmax_3;  % e.g. 24
elseif isfield(css_data, 'bmax')
    bmax = css_data.bmax;
else
    bmax = 24;
end

Bs_old= css_data.Bs;      % Original steady state bond supply

% New constraint
bmin_new = -2.5;
bgrid_new = linspace(bmin_new, bmax, Nb)';

%% Dist at SS prices
disp('Computing B_s''...');

% Set up grid and transitions using original steady-state values
ygrid_fixed = [w_css, z];
Pi_fixed = [1-sigma, sigma; f_css, 1-f_css];

% Initial guess for the policy function
bp_old = r_css * [bgrid_new, bgrid_new];

% Household EGM loop
dist_val = 10;
iterhh = 0;
while dist_val > 1e-6 && iterhh < 2000
    iterhh = iterhh + 1;
    [bp_new, ~] = egmUpdateSS(bp_old, bgrid_new, ygrid_fixed, Pi_fixed, beta, r_css, gamma, bmin_new);
    dist_val = max(abs(bp_new(:) - bp_old(:)));
    bp_old = bp_new;
end
%% Find new B_s'
% Simulate stationary distribution using Young's (2010) method
g_ss_new = solveErgDist(bp_new, bgrid_new, f_css, sigma);

% Calculate the new aggregate demand for bonds
Bd_new = sum(bgrid_new .* g_ss_new, 'all');

Bs_prime = Bd_new;

fprintf('B_s'': %.6f\n', Bs_prime);

%% ================================================================
%  FIGURES FOR Q3.2 -- Fiscal Policy Stabilisation
%  ================================================================

bgrid_css = css_data.bgrid_3;
g_ss_css  = css_data.g_ss_3;

col_css = [0.20 0.45 0.70];
col_new = [0.18 0.63 0.18];   % green tones for fiscal

%% Figure 1: Stationary Distribution Overlay
figure('Name','Q3.2 -- Distribution Overlay','NumberTitle','off');
set(gcf,'Position',[80 80 960 440]);

subplot(1,2,1); hold on;
area(bgrid_css, g_ss_css(:,1),'FaceColor',col_css,'FaceAlpha',0.50,'EdgeColor',col_css,'LineWidth',1.5,'DisplayName','Calibrated SS');
area(bgrid_new, g_ss_new(:,1),'FaceColor',col_new,'FaceAlpha',0.40,'EdgeColor',col_new,'LineWidth',1.5,'DisplayName','Fiscal Stab.\ ($B_s''$)');
xline(-3,   ':','Color',[0.5 0.5 0.5],'LineWidth',1.1,'HandleVisibility','off');
xline(-2.5,'--k','LineWidth',1.2,'HandleVisibility','off');
text(-2.4, max(g_ss_css(:,1))*0.55,'$b_{min}''$','Interpreter','latex','FontSize',9);
xlabel('Bond holdings $b$'); ylabel('Prob.\ mass $g(b,e)$');
title('Employed Households ($x=e$)'); legend('Location','best','FontSize',8);
xlim([-3.5 8]); grid on; box on;

subplot(1,2,2); hold on;
area(bgrid_css, g_ss_css(:,2),'FaceColor',col_css,'FaceAlpha',0.50,'EdgeColor',col_css,'LineWidth',1.5,'DisplayName','Calibrated SS');
area(bgrid_new, g_ss_new(:,2),'FaceColor',col_new,'FaceAlpha',0.40,'EdgeColor',col_new,'LineWidth',1.5,'DisplayName','Fiscal Stab.\ ($B_s''$)');
xline(-3,   ':','Color',[0.5 0.5 0.5],'LineWidth',1.1,'HandleVisibility','off');
xline(-2.5,'--k','LineWidth',1.2,'HandleVisibility','off');
text(-2.4, max(g_ss_css(:,2))*0.55,'$b_{min}''$','Interpreter','latex','FontSize',9);
xlabel('Bond holdings $b$'); ylabel('Prob.\ mass $g(b,u)$');
title('Unemployed Households ($x=u$)'); legend('Location','best','FontSize',8);
xlim([-3.5 8]); grid on; box on;

sgtitle('Q3.2 -- Stationary Distribution: SS vs.\ Fiscal Stabilisation','FontWeight','bold');

%% Figure 2: Bond Market -- Supply Shift
fprintf('\n  Sweeping r to build bond demand curve (Q3.2 bond market)...\n');

ygrid_fixed = [w_css, z];
Pi_fixed    = [1-sigma, sigma; f_css, 1-f_css];
r_sweep     = linspace(1.0010, 1.0040, 30);
Bd_sweep    = zeros(1,30);

for k = 1:30
    r_k   = r_sweep(k);
    bp_sw = r_k * [bgrid_new, bgrid_new];
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

figure('Name','Q3.2 -- Bond Market (Supply Shift)','NumberTitle','off');
set(gcf,'Position',[200 80 660 500]);
hold on;

% Bond demand curve
plot(Bd_sweep, r_sweep, 'b-', 'LineWidth', 2.5, 'DisplayName', '$B^d(r)$, new $b_{min}=-2.5$');

% Old supply (vertical)
plot([Bs_old Bs_old], [r_sweep(1) r_sweep(end)], '--', 'Color',[0.5 0.5 0.5], 'LineWidth', 2, 'DisplayName', '$B^s$ (original)');

% New supply (vertical) -- the fiscal intervention
plot([Bs_prime Bs_prime], [r_sweep(1) r_sweep(end)], '-', 'Color',[0.13 0.55 0.13], 'LineWidth', 2.5, 'DisplayName', '$B_s''$ (fiscal expansion)');

% Reference rate r_ss
yline(r_css, '--', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.4, 'HandleVisibility','off');
text(min(Bd_sweep)+0.01, r_css+0.00008,'$r_{ss}$','Interpreter','latex','FontSize',11,'Color',[0.85 0.33 0.10]);

% Equilibrium markers
scatter(Bs_old,  r_css, 90, [0.5 0.5 0.5], 'o','LineWidth',1.5, 'DisplayName','Old equilibrium $(B^s,\,r_{ss})$');
scatter(Bs_prime, r_css, 110, [0.13 0.55 0.13], 'filled','MarkerEdgeColor','k','DisplayName','New equilibrium $(B_s'',\,r_{ss})$');

xlabel('Bond Quantity $B$'); ylabel('Real Interest Rate $r$');
title('Q3.2 -- Bond Market: Fiscal Supply Expansion at $r_{ss}$');
legend('Location','southeast','FontSize',9);
grid on; box on;
