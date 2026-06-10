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


%% 3.c Zero Lower Bound
css_data = load('ws_steady.mat');

w_css = css_data.w;
beta  = css_data.beta;
sigma = css_data.sigma;
f_css = css_data.f;
z     = css_data.z;
gamma = css_data.gamma;
Nb    = css_data.Nb;
bmax  = css_data.bmax_3;  % bmax = 24
Bs    = css_data.Bs;      % Steady state bond supply

% Inflation target
pi_ss  = 1 + (0.02/12);

%% r_zlb

r_zlb = 1 / pi_ss;

disp('Finding b_min'' at ZLB...');

%% Loop for b_min'

ygrid_fixed = [w_css, z];
Pi_fixed = [1-sigma, sigma; f_css, 1-f_css];

% Initial guess for the financial shock
% Since r_zlb is lower than r_ss, we need more demand, so bmin must be tighter than -3
bmin_guess = -2.5; 
% We can use a simple secant method or gradient descent
tol_bmin = 1e-4;
err = 10;
lambda_bmin = 0.5; % Step size for the update
max_iter = 100;
iter = 0;

% Step size for the update

while abs(err) > tol_bmin && iter < max_iter
    iter = iter + 1;
    
    % 1. Update grid with the new guessed bmin
    bgrid_new = linspace(bmin_guess, bmax, Nb)';
    
    % 2. Solve EGM for household policies at r_zlb
    bp_old = r_zlb * [bgrid_new, bgrid_new];
    dist_val = 10; iterhh = 0;
    while dist_val > 1e-6 && iterhh < 2000
        iterhh = iterhh + 1;
        [bp_new, ~] = egmUpdateSS(bp_old, bgrid_new, ygrid_fixed, Pi_fixed, beta, r_zlb, gamma, bmin_guess);
        dist_val = max(abs(bp_new(:) - bp_old(:)));
        bp_old = bp_new;
    end
    
    % 3. Compute stationary distribution
    g_ss_zlb = solveErgDist(bp_new, bgrid_new, f_css, sigma);
    
    % 4. Aggregate bond demand
    Bd_zlb = sum(bgrid_new .* g_ss_zlb, 'all');
    
    err = Bs - Bd_zlb;
    
    bmin_guess = bmin_guess + lambda_bmin * err;
    
    fprintf('Iter %d: b_min_guess = %.4f, Bd = %.4f, err = %.4f\n', iter, bmin_guess, Bd_zlb, err);
end

fprintf('b_min'': %.6f\n', bmin_guess);

%% ================================================================
%  FIGURES FOR Q3.3 -- Zero Lower Bound
%  ================================================================

r_css    = css_data.r;
bgrid_css = css_data.bgrid_3;
g_ss_css  = css_data.g_ss_3;

col_css = [0.20 0.45 0.70];   % blue   (SS)
col_zlb = [0.60 0.10 0.80];   % purple (ZLB shock)

%% Figure 1: Stationary Distribution Overlay  (SS vs ZLB shock)
figure('Name','Q3.3 -- Distribution Overlay','NumberTitle','off');
set(gcf,'Position',[80 80 960 440]);

subplot(1,2,1); hold on;
area(bgrid_css,  g_ss_css(:,1), 'FaceColor',col_css,'FaceAlpha',0.50,'EdgeColor',col_css,'LineWidth',1.5,'DisplayName','Calibrated SS ($b_{min}=-3$)');
area(bgrid_new,  g_ss_zlb(:,1),'FaceColor',col_zlb,'FaceAlpha',0.40,'EdgeColor',col_zlb,'LineWidth',1.5,'DisplayName',sprintf('ZLB Shock ($b_{min}''=%.3f$)',bmin_guess));
xline(-3,         ':','Color',[0.5 0.5 0.5],'LineWidth',1.1,'HandleVisibility','off');
xline(bmin_guess,'--k','LineWidth',1.2,'HandleVisibility','off');
text(bmin_guess+0.02, max(g_ss_css(:,1))*0.55,'$b_{min}''$','Interpreter','latex','FontSize',9);
xlabel('Bond holdings $b$'); ylabel('Prob.\ mass $g(b,e)$');
title('Employed Households ($x=e$)'); legend('Location','best','FontSize',8);
xlim([-3.3 8]); grid on; box on;

subplot(1,2,2); hold on;
area(bgrid_css,  g_ss_css(:,2),'FaceColor',col_css,'FaceAlpha',0.50,'EdgeColor',col_css,'LineWidth',1.5,'DisplayName','Calibrated SS ($b_{min}=-3$)');
area(bgrid_new,  g_ss_zlb(:,2),'FaceColor',col_zlb,'FaceAlpha',0.40,'EdgeColor',col_zlb,'LineWidth',1.5,'DisplayName',sprintf('ZLB Shock ($b_{min}''=%.3f$)',bmin_guess));
xline(-3,         ':','Color',[0.5 0.5 0.5],'LineWidth',1.1,'HandleVisibility','off');
xline(bmin_guess,'--k','LineWidth',1.2,'HandleVisibility','off');
text(bmin_guess+0.02, max(g_ss_css(:,2))*0.55,'$b_{min}''$','Interpreter','latex','FontSize',9);
xlabel('Bond holdings $b$'); ylabel('Prob.\ mass $g(b,u)$');
title('Unemployed Households ($x=u$)'); legend('Location','best','FontSize',8);
xlim([-3.3 8]); grid on; box on;

sgtitle('Q3.3 -- Stationary Distribution: SS vs.\ ZLB Financial Shock','FontWeight','bold');

%% Figure 2: Bond Market at ZLB  (wider r sweep, 3 reference lines)
fprintf('\n  Sweeping r for bond market figure (Q3.3)...\n');

r_sw_lo = 0.9970;  r_sw_hi = 1.0040;
r_sweep_zlb  = linspace(r_sw_lo, r_sw_hi, 30);
Bd_sweep_zlb = zeros(1,30);

% Demand uses ZLB bmin and ZLB grid (bmin_guess)
bgrid_zlb_sw = bgrid_new;    % already linspace(bmin_guess, bmax, Nb)

for k = 1:30
    r_k   = r_sweep_zlb(k);
    bp_sw = r_k * [bgrid_zlb_sw, bgrid_zlb_sw];
    d_sw  = 10;  it_sw = 0;
    while d_sw > 1e-6 && it_sw < 2500
        it_sw = it_sw + 1;
        [bp_sw_new, ~] = egmUpdateSS(bp_sw, bgrid_zlb_sw, ygrid_fixed, Pi_fixed, beta, r_k, gamma, bmin_guess);
        d_sw  = max(abs(bp_sw_new(:) - bp_sw(:)));
        bp_sw = bp_sw_new;
    end
    g_sw             = solveErgDist(bp_sw, bgrid_zlb_sw, f_css, sigma);
    Bd_sweep_zlb(k)  = sum(bgrid_zlb_sw .* g_sw, 'all');
end

figure('Name','Q3.3 -- Bond Market at ZLB','NumberTitle','off');
set(gcf,'Position',[200 80 680 520]);
hold on;

r_star_q31 = 1.002156;   % natural rate from Q3.1 (hardcoded from results)

plot(Bd_sweep_zlb, r_sweep_zlb, '-', 'Color',col_zlb, 'LineWidth',2.5, 'DisplayName','$B^d(r)$, ZLB shock');
plot([Bs Bs], [r_sw_lo r_sw_hi], '-', 'Color',[0.13 0.55 0.13], 'LineWidth',2.5, 'DisplayName','$B^s$ (fixed supply)');

yline(r_css,      '--','Color',[0.5  0.5  0.5 ],'LineWidth',1.4,'HandleVisibility','off');
yline(r_star_q31, ':','Color',[0.85 0.33 0.10],'LineWidth',1.4,'HandleVisibility','off');
yline(r_zlb,      '-.','Color',col_zlb,'LineWidth',1.4,'HandleVisibility','off');

Bd_xmin = min(Bd_sweep_zlb);
text(Bd_xmin+0.005, r_css+0.00010,      '$r_{ss}$',        'Interpreter','latex','FontSize',11,'Color',[0.5 0.5 0.5]);
text(Bd_xmin+0.005, r_star_q31-0.00015, '$r^*$ (Q3.1)',    'Interpreter','latex','FontSize',10,'Color',[0.85 0.33 0.10]);
text(Bd_xmin+0.005, r_zlb-0.00015,      '$r_{zlb}=1/\pi_{ss}$','Interpreter','latex','FontSize',10,'Color',col_zlb);

% Shade the monetary policy space  (r_zlb to r_css)
patch([min(Bd_sweep_zlb) max(Bd_sweep_zlb) max(Bd_sweep_zlb) min(Bd_sweep_zlb)], ...
      [r_zlb r_zlb r_css r_css], [0.9 0.95 0.9], 'FaceAlpha',0.3,'EdgeColor','none','DisplayName','Monetary policy space');

scatter(Bs, r_zlb, 110, col_zlb, 'filled','MarkerEdgeColor','k','DisplayName','ZLB equilibrium');

xlabel('Bond Quantity $B$'); ylabel('Real Interest Rate $r$');
title('Q3.3 -- Bond Market: ZLB Constraint with Policy Space');
legend('Location','southeast','FontSize',9);
grid on; box on;

%% Figure 3: ZLB Frontier  --  r*(b_min) for b_min in [-3, bmin_guess]
fprintf('\n  Computing ZLB frontier curve r*(b_min)...\n');

bmin_frontier = linspace(-3, bmin_guess, 20);
rstar_frontier = zeros(1,20);

for j = 1:20
    bm_j     = bmin_frontier(j);
    bgrid_j  = linspace(bm_j, bmax, Nb)';
    r_j      = r_css;      % start guess
    err_j    = 10;  iter_j = 0;
    bp_j     = r_j * [bgrid_j, bgrid_j]; % Init outside the while loop!
    while abs(err_j) > 1e-4 && iter_j < 500
        iter_j = iter_j + 1;
        d_j    = 10;  ith = 0;
        while d_j > 1e-6 && ith < 2500
            ith = ith + 1;
            [bp_j_new, ~] = egmUpdateSS(bp_j, bgrid_j, ygrid_fixed, Pi_fixed, beta, r_j, gamma, bm_j);
            d_j  = max(abs(bp_j_new(:) - bp_j(:)));
            bp_j = bp_j_new;
        end
        g_j   = solveErgDist(bp_j, bgrid_j, f_css, sigma);
        Bd_j  = sum(bgrid_j .* g_j, 'all');
        err_j = Bs - Bd_j;
        r_j   = r_j + 0.001 * err_j;
    end
    rstar_frontier(j) = r_j;
    fprintf('    b_min = %.3f  ->  r* = %.6f\n', bm_j, r_j);
end

figure('Name','Q3.3 -- ZLB Frontier','NumberTitle','off');
set(gcf,'Position',[300 80 680 500]);
hold on;

plot(bmin_frontier, rstar_frontier, 'b-o', 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor','b', 'DisplayName', '$r^*(b_{min}'')$');
yline(r_css,  '--', 'Color',[0.5 0.5 0.5], 'LineWidth',1.4, 'HandleVisibility','off');
yline(r_zlb,  '-.', 'Color',col_zlb,       'LineWidth',1.4, 'HandleVisibility','off');
xline(bmin_guess,'--k','LineWidth',1.2,'HandleVisibility','off');

text(-2.95, r_css+0.00008,      '$r_{ss}$ (upper bound)',      'Interpreter','latex','FontSize',10,'Color',[0.5 0.5 0.5]);
text(-2.95, r_zlb-0.00015,      '$r_{zlb}=1/\pi_{ss}$ (ZLB)', 'Interpreter','latex','FontSize',10,'Color',col_zlb);
text(bmin_guess+0.02, (r_css+r_zlb)/2, {'ZLB', 'trigger'},'Interpreter','latex','FontSize',9,'HorizontalAlignment','left');

% Shade the accommodable region
patch([bmin_frontier(1) bmin_frontier(end) bmin_frontier(end) bmin_frontier(1)], ...
      [r_zlb r_zlb r_css r_css], [0.9 0.95 0.9],'FaceAlpha',0.35,'EdgeColor','none','DisplayName','Monetary policy space');

xlabel('Borrowing Constraint $b_{min}''$'); ylabel('Market-clearing Real Rate $r^*$');
title('Q3.3 -- ZLB Frontier: Natural Rate as a Function of the Borrowing Constraint');
legend('Location','northeast','FontSize',9);
grid on; box on;

% Save the plot for LaTeX compilation
exportgraphics(gcf, '../figures/Q33_ZLB_frontier.png', 'Resolution', 300);

%% Figure 4: 4-Panel Cross-Scenario Distribution Comparison
fprintf('\n  Computing Q3.1 and Q3.2 distributions for 4-panel comparison...\n');

% --- Q3.1 distribution: r_star_q31, bmin=-2.5, labour at SS ---
bmin_31   = -2.5;
bgrid_31  = linspace(bmin_31, bmax, Nb)';
bp_31     = r_star_q31 * [bgrid_31, bgrid_31];
d_31      = 10;  it31 = 0;
while d_31 > 1e-6 && it31 < 2500
    it31 = it31 + 1;
    [bp_31_new, ~] = egmUpdateSS(bp_31, bgrid_31, ygrid_fixed, Pi_fixed, beta, r_star_q31, gamma, bmin_31);
    d_31   = max(abs(bp_31_new(:) - bp_31(:)));
    bp_31  = bp_31_new;
end
g_31 = solveErgDist(bp_31, bgrid_31, f_css, sigma);

% --- Q3.2 distribution: r_css, bmin=-2.5, labour at SS ---
bp_32 = r_css * [bgrid_31, bgrid_31];
d_32  = 10;  it32 = 0;
while d_32 > 1e-6 && it32 < 2500
    it32 = it32 + 1;
    [bp_32_new, ~] = egmUpdateSS(bp_32, bgrid_31, ygrid_fixed, Pi_fixed, beta, r_css, gamma, bmin_31);
    d_32  = max(abs(bp_32_new(:) - bp_32(:)));
    bp_32 = bp_32_new;
end
g_32 = solveErgDist(bp_32, bgrid_31, f_css, sigma);

% Colours
col_q31 = [0.85 0.33 0.10];   % orange (monetary)
col_q32 = [0.18 0.63 0.18];   % green  (fiscal)

% Shared x-axis limits for comparability
xlims4 = [-3.3 8];

figure('Name','Q3 -- 4-Panel Distribution Comparison','NumberTitle','off');
set(gcf,'Position',[100 60 960 800]);

% Panel 1: Calibrated SS
subplot(2,2,1); hold on;
area(bgrid_css, g_ss_css(:,1),'FaceColor',col_css,'FaceAlpha',0.55,'EdgeColor',col_css,'LineWidth',1.4,'DisplayName','Employed ($x=e$)');
area(bgrid_css, g_ss_css(:,2),'FaceColor',[0.9 0.6 0.1],'FaceAlpha',0.45,'EdgeColor',[0.7 0.45 0.0],'LineWidth',1.4,'DisplayName','Unemployed ($x=u$)');
xline(-3,'--k','LineWidth',1,'HandleVisibility','off');
xlabel('$b$'); ylabel('Prob.\ mass'); title('Calibrated SS ($b_{min}=-3$)');
legend('Location','best','FontSize',8); xlim(xlims4); grid on; box on;

% Panel 2: Q3.1 Monetary
subplot(2,2,2); hold on;
area(bgrid_31, g_31(:,1),'FaceColor',col_q31,'FaceAlpha',0.55,'EdgeColor',col_q31,'LineWidth',1.4,'DisplayName','Employed ($x=e$)');
area(bgrid_31, g_31(:,2),'FaceColor',[0.9 0.6 0.1],'FaceAlpha',0.45,'EdgeColor',[0.7 0.45 0.0],'LineWidth',1.4,'DisplayName','Unemployed ($x=u$)');
xline(-2.5,'--k','LineWidth',1,'HandleVisibility','off');
xlabel('$b$'); ylabel('Prob.\ mass'); title(sprintf('Q3.1 -- Monetary Stab.\\ ($\\epsilon^*=%.4f, r^*=%.4f$)', 0.999657, r_star_q31));
legend('Location','best','FontSize',8); xlim(xlims4); grid on; box on;

% Panel 3: Q3.2 Fiscal
subplot(2,2,3); hold on;
area(bgrid_31, g_32(:,1),'FaceColor',col_q32,'FaceAlpha',0.55,'EdgeColor',col_q32,'LineWidth',1.4,'DisplayName','Employed ($x=e$)');
area(bgrid_31, g_32(:,2),'FaceColor',[0.9 0.6 0.1],'FaceAlpha',0.45,'EdgeColor',[0.7 0.45 0.0],'LineWidth',1.4,'DisplayName','Unemployed ($x=u$)');
xline(-2.5,'--k','LineWidth',1,'HandleVisibility','off');
xlabel('$b$'); ylabel('Prob.\ mass'); title('Q3.2 -- Fiscal Stab.\ ($B_s''=3.994, r=r_{ss}$)');
legend('Location','best','FontSize',8); xlim(xlims4); grid on; box on;

% Panel 4: Q3.3 ZLB
subplot(2,2,4); hold on;
area(bgrid_new, g_ss_zlb(:,1),'FaceColor',col_zlb,'FaceAlpha',0.55,'EdgeColor',col_zlb,'LineWidth',1.4,'DisplayName','Employed ($x=e$)');
area(bgrid_new, g_ss_zlb(:,2),'FaceColor',[0.9 0.6 0.1],'FaceAlpha',0.45,'EdgeColor',[0.7 0.45 0.0],'LineWidth',1.4,'DisplayName','Unemployed ($x=u$)');
xline(bmin_guess,'--k','LineWidth',1,'HandleVisibility','off');
xlabel('$b$'); ylabel('Prob.\ mass'); title(sprintf('Q3.3 -- ZLB Shock ($b_{min}''=%.3f, r_{zlb}=%.4f$)', bmin_guess, r_zlb));
legend('Location','best','FontSize',8); xlim(xlims4); grid on; box on;

sgtitle('Q3 -- Cross-Scenario Comparison of Stationary Distributions $g(b,x)$','FontWeight','bold');

