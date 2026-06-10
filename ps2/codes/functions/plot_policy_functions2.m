function plot_policy_functions2(bgrid, bp, c)
    % PLOT_POLICY_FUNCTIONS2 Generates and saves steady-state plots for bmax=10.
    
    % Ensure the figures directory exists relative to the codes folder
    fig_dir = '../figures/';
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    % Set Font Sizes
    base_fs = 20; % Large font size for readability
    label_fs = 22;
    title_fs = 24;

    % Set LaTeX interpreters globally
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');
    set(groot, 'defaultTextInterpreter', 'latex');

    % 1. Plot Savings Policy Function b'(b,x)
    fig_bp = figure('Name', 'Savings Policy (bmax=10)', 'Color', 'w', 'Visible', 'off');
    ax1 = axes('FontSize', base_fs); % Sets tick label size
    hold on;
    plot(bgrid, bp(:, 1), 'LineWidth', 2.5, 'DisplayName', 'Employed');
    plot(bgrid, bp(:, 2), 'LineWidth', 2.5, 'DisplayName', 'Unemployed');
    plot(bgrid, bgrid, 'k--', 'LineWidth', 1.5, 'DisplayName', '45$^\circ$');
    
    % Updated Title
    title('Savings Policy Function $b''(b, x)$ ($b_{max}=10$)', 'FontSize', title_fs);
    xlabel('Current Assets $b$', 'FontSize', label_fs);
    ylabel('Next Period Assets $b''$', 'FontSize', label_fs);
    
    legend('Location', 'northwest', 'FontSize', base_fs);
    grid on;
    hold off;

    % Updated Filename
    saveas(fig_bp, fullfile(fig_dir, 'Q1_savings_policy_bmax10.png'));

    % 2. Plot Consumption Policy Function c(b,x)
    fig_c = figure('Name', 'Consumption Policy (bmax=10)', 'Color', 'w', 'Visible', 'off');
    ax2 = axes('FontSize', base_fs); % Sets tick label size
    hold on;
    plot(bgrid, c(:, 1), 'LineWidth', 2.5, 'DisplayName', 'Employed');
    plot(bgrid, c(:, 2), 'LineWidth', 2.5, 'DisplayName', 'Unemployed');
    
    % Updated Title
    title('Consumption Policy Function $c(b, x)$ ($b_{max}=10$)', 'FontSize', title_fs);
    xlabel('Current Assets $b$', 'FontSize', label_fs);
    ylabel('Consumption $c$', 'FontSize', label_fs);
    
    legend('Location', 'northwest', 'FontSize', base_fs);
    grid on;
    hold off;

    % Updated Filename
    saveas(fig_c, fullfile(fig_dir, 'Q1_consumption_policy_bmax10.png'));
end