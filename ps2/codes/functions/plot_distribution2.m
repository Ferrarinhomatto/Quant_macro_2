function plot_distribution2(bgrid, g_ss)
    % PLOT_DISTRIBUTION2 Generates and saves the stationary distribution plot (bmax=10).
    
    % Ensure the figures directory exists relative to the codes folder
    fig_dir = '../figures/';
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    % Set Font Sizes for high readability
    base_fs = 20; 
    label_fs = 22;
    title_fs = 24;

    % Set LaTeX interpreters globally
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');
    set(groot, 'defaultTextInterpreter', 'latex');

    % Create the figure
    fig_dist = figure('Name', 'Stationary Distribution (bmax=10)', 'Color', 'w', 'Visible', 'off');
    ax = axes('FontSize', base_fs);
    hold on;
    
    % Plot the probability mass function for each state
    plot(bgrid, g_ss(:, 1), 'LineWidth', 2.5, 'DisplayName', 'Employed');
    plot(bgrid, g_ss(:, 2), 'LineWidth', 2.5, 'DisplayName', 'Unemployed');

    % Formatting
    title('Stationary Asset Distribution $g(b, x)$ ($b_{max}=10$)', 'FontSize', title_fs);
    xlabel('Assets $b$', 'FontSize', label_fs);
    ylabel('Probability Mass', 'FontSize', label_fs);
    
    legend('Location', 'northeast', 'FontSize', base_fs);
    grid on;
    hold off;

    % Save the figure
    saveas(fig_dist, fullfile(fig_dir, 'Q1_asset_distribution_bmax10.png'));
end