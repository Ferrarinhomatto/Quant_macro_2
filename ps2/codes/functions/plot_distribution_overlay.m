function plot_distribution_overlay(bgrid1, g1, bgrid2, g2, filename)
    f = figure('Visible', 'off');
    
    % Original SS (The Baseline: Black and Gray)
    plot(bgrid1, g1(:,1), 'k-', 'LineWidth', 2); hold on;
    plot(bgrid1, g1(:,2), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 2);
    
    % New SS (The Shock: Bright Blue and Red)
    plot(bgrid2, g2(:,1), 'b-', 'LineWidth', 2);
    plot(bgrid2, g2(:,2), 'r--', 'LineWidth', 2);
    
    % --- Calculate Conditional Means ---
    mean_e1 = sum(bgrid1 .* g1(:,1)) / sum(g1(:,1));
    mean_u1 = sum(bgrid1 .* g1(:,2)) / sum(g1(:,2));
    mean_e2 = sum(bgrid2 .* g2(:,1)) / sum(g2(:,1));
    mean_u2 = sum(bgrid2 .* g2(:,2)) / sum(g2(:,2));
    
    % --- Plot Vertical Lines for Means ---
    % Using matching colors and dotted lines (':')
    xline(mean_e1, 'k:', 'LineWidth', 2, 'HandleVisibility', 'off');
    xline(mean_u1, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'HandleVisibility', 'off');
    xline(mean_e2, 'b:', 'LineWidth', 2, 'HandleVisibility', 'off');
    xline(mean_u2, 'r:', 'LineWidth', 2, 'HandleVisibility', 'off');
    
    % Global font sizing for readability
    set(gca, 'FontSize', 14);
    
    title('Asset Distribution Shift ($b_{min} \rightarrow b_{min}^{\prime}$)', ...
          'Interpreter', 'latex', 'FontSize', 18);
    xlabel('Assets ($b$)', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('Probability Mass $g(b,x)$', 'Interpreter', 'latex', 'FontSize', 16);
    
    legend('Emp (Original)', 'Unemp (Original)', 'Emp (New)', 'Unemp (New)', ...
           'Location', 'best', 'Interpreter', 'latex', 'FontSize', 12);
    grid on;
    
    % Save and close
    saveas(f, fullfile('../figures/', [filename, '.png']));
    close(f);
end