function solving_EGM = EGM_het(par, cpar, gri)
% EGM_HET: Endogenous Gridpoints Method for Heterogeneous Returns
    % Ensure inputs are column vectors
    gri.k = gri.k(:);
    gri.z = gri.z(:);
    gri.r = gri.r(:);
    
    % Global grid bounds for cash-on-hand
    x_min = min(gri.z)*par.w + min(1 + par.r_bar + gri.r)*gri.k(1);
    x_max = max(gri.z)*par.w + max(1 + par.r_bar + gri.r)*gri.k(end);
        gri.x = linspace(x_min, x_max, 100)';

    
        for j = 1:cpar.N_het
        kpol_old(:, j) = 0.5 * gri.x;
    end
    
    diff = cpar.tol + 1; iter = 0;
    
    while diff > cpar.tol && iter < cpar.maxit
        kpol_new = zeros(size(kpol_old));
        
        for j = 1:cpar.N_het
            c_tod = zeros(cpar.Nkap, 1);
            x_end = zeros(cpar.Nkap, 1);
            
            for kp = 1:cpar.Nkap
                exp_MU = 0;
                for m = 1:cpar.N_het
                    prob_trans = gri.prob(j, m);
                    if prob_trans > 0
                        % Tomorrow's stochastic return is inside the expectation!
                        R_tom = 1 + par.r_bar + gri.r(m);
                        x_tom = gri.z(m) * par.w + R_tom * gri.k(kp);
                        
                        k_tom_tom = interp1(gri.x, kpol_old(:,m), x_tom, 'linear', 'extrap');
                        k_tom_tom = max(0, min(k_tom_tom, x_tom - 1e-10));
                        c_tom = x_tom - k_tom_tom;
                        
                        % Euler Equation with stochastic R
                        exp_MU = exp_MU + prob_trans * R_tom * (c_tom^(-par.sigma));
                    end
                end
                c_tod(kp) = (par.beta * exp_MU)^(-1/par.sigma);
                x_end(kp) = gri.k(kp) + c_tod(kp);
            end
            
                        x_end_bc = [0; x_end];
            k_with_BC = [0; gri.k];
            
                        kpol_new(:, j) = interp1(x_end_bc, k_with_BC, gri.x, "linear","extrap");
            kpol_new(:, j) = max(0, min(kpol_new(:, j), gri.x));
        end
        diff = max(abs(kpol_new - kpol_old), [], 'all');
        kpol_old = kpol_new;
        iter = iter + 1;
    end
    
    % Store the final x-policy
    solving_EGM.kpol_x = kpol_new;
    
    % Map back from (x, state) space to standard (k, state) space
    solving_EGM.kpol_k = zeros(cpar.Nkap, cpar.N_het);
        for j = 1:cpar.N_het
        R_today = 1 + par.r_bar + gri.r(j);
        x_today = gri.z(j) * par.w + R_today * gri.k;
        solving_EGM.kpol_k(:, j) = interp1(gri.x, kpol_new(:, j), x_today, 'linear', 'extrap');
    end
    
    % Enforce constraints
    solving_EGM.kpol_k = max(0, min(solving_EGM.kpol_k, max(gri.k)));
end