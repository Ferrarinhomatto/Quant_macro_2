function plot_distribution_simple(bgrid, g_ss, filename, title_str)
    f = figure('Visible', 'off');
    
    plot(bgrid, g_ss(:,1), 'b-', 'LineWidth', 2); hold on;
    plot(bgrid, g_ss(:,2), 'r--', 'LineWidth', 2);
    
    set(gca, 'FontSize', 12);
    title(title_str, 'Interpreter', 'latex', 'FontSize', 14);
    xlabel('Assets ($b$)', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('Mass $g(b,x)$', 'Interpreter', 'latex', 'FontSize', 14);
    legend('Employed', 'Unemployed', 'Location', 'best', 'Interpreter', 'latex');
    grid on;
    
    saveas(f, fullfile('../figures/', [filename, '.png']));
    close(f);
end