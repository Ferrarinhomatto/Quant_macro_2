function euler_errors = EErrors(V, cpol, kpol, npol, ipol, gri, par)
    euler_errors = zeros(par.M, par.Na); % Initializing storing matrix
    mu = @(c) c.^(-par.gamma); % Marginal Utility of C
    
    for i_a = 1:par.M
    for i_k = 1:par.Na
        % Current states
        current_C = cpol(i_a, i_k);
        current_I = ipol(i_a, i_k);
        current_K = gri.kap(i_k);
        
        % Ignore C<0
        if current_C < 1e-10
            euler_errors(i_a, i_k) = NaN;
            continue;
        end
        
        % Current It/Kt (needed in Euler Equation, which is the following,
        % taking into account that ):
        % LHS = u'(C_t) / [1 - Φ(I_t/K_t - δ)]
        % 
        % RHS = β E_t[ u'(C_{t+1}) * R_{t+1} / [1 - Φ(I_{t+1}/K_{t+1} - δ)] ]
        % 
        % Where:
        %   R_{t+1} = MPK_{t+1} + 1 - δ + ADJ_{t+1}
        %   ADJ_{t+1} = [ (Φ·δ·(I_{t+1}/K_{t+1})/2) - (Φ·δ²/2) + (1-δ) ] 
        %                / [ 1 - Φ((I_{t+1}/K_{t+1}) - δ) ]

        inv_rate_t0 = current_I/current_K; % Investment rate today...
        % needed in lhs

        mu_t = mu(current_C);
        
        % LHS with adjustment cost division
        lhs = mu_t/(1-(par.phii*(inv_rate_t0-par.delta)));
        
        expected_rhs = 0; % Initializing rhs
        valid_transitions = 0; % Transitions auxiliar
        
        for j_a = 1:par.M
            prob = gri.prob(i_a, j_a);
            % Interpolating for tomorrow's states
            K_t1 = kpol(i_a, i_k);
            C_t1 = interp1(gri.kap, cpol(j_a, :), K_t1, 'linear', 'extrap');
            N_t1 = interp1(gri.kap, npol(j_a, :), K_t1, 'linear', 'extrap');
            I_t1 = interp1(gri.kap, ipol(j_a, :), K_t1, 'linear', 'extrap');
            
            if C_t1 < 1e-10, continue; end % Ignore if C<0
            
            inv_rate_t1 = I_t1 / K_t1; % Inv rate tomorrow
            MPK_t1 = par.alpha * gri.A(j_a) * K_t1^(par.alpha-1) * N_t1^(1-par.alpha); % MPK
            
            if par.phii == 0
                adj_terms = 1-par.delta; % Add to rate of return of capital when...
                % adjustment costs are zero
            else
                adj_terms = ((par.phii*par.delta*inv_rate_t1/2)-...
                    (par.phii*par.delta^2/2)+(1-par.delta))/(1-(par.phii*(inv_rate_t1-par.delta)));
                % Add to rate of return of capital otherwise
            end
            
            R_t1 = MPK_t1 + adj_terms; % Rate of retrun of capital given phi
            mu_t1 = mu(C_t1); % MgU of C tomorrow
            expected_rhs = expected_rhs + prob * mu_t1 * R_t1;
            % Expected RHS
            valid_transitions = valid_transitions + prob;
        end
        
        rhs = par.beta * expected_rhs; % Final RHS (multiplied by discount factor)
        
        % Absolute Euler Error (works even if rhs is negative)
        if valid_transitions > 0.1 && abs(lhs) > 1e-10
            absolute_error = abs(lhs - rhs);
            euler_errors(i_a, i_k) = log10(absolute_error);  % log of absolute error
        else
            euler_errors(i_a, i_k) = NaN;
        end
    end
    end
end