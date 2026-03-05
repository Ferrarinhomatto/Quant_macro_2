function solving_EGM = EGM_het(par, cpar, gri)
% EGM_HET: Endogenous Gridpoints Method for Heterogeneous Returns
%
% VECTORIZED VERSION with:
%   - Log-spaced cash-on-hand grid (matches log-spaced k grid)
%   - No artificial upper-bound clamping on k' (only borrowing constraint)

    gri.k = gri.k(:);
    gri.z = gri.z(:);
    gri.r = gri.r(:);
    
    N_het = cpar.N_het;  % = 25 typically
    Nkap = cpar.Nkap;
    
    % Global grid bounds for cash-on-hand
    x_min = min(gri.z)*par.w + min(1 + par.r_bar + gri.r)*gri.k(1);
    x_max = max(gri.z)*par.w + max(1 + par.r_bar + gri.r)*gri.k(end);
    
    % Log-spaced x grid: better resolution near borrowing constraint
    Nx = 200;
    gri.x = exp(linspace(log(x_min + 1), log(x_max + 1), Nx))' - 1;

    % Initial guess: save half of cash-on-hand
    kpol_old = 0.5 * repmat(gri.x, 1, N_het);
    
    diff = cpar.tol + 1; 
    iter = 0;
    
    % Pre-compute tomorrow's return and cash-on-hand for each (kp, m) pair
    R_tom = (1 + par.r_bar + gri.r)';            % 1 x N_het
    x_tom_mat = gri.k * R_tom + par.w * gri.z';  % Nkap x N_het
    
    while diff > cpar.tol && iter < cpar.maxit
        kpol_new = zeros(size(kpol_old));
        
        % --- Step 1: Batch-interpolate tomorrow's policy for all future states ---
        k_tom_tom = zeros(Nkap, N_het);
        for m = 1:N_het
            k_tom_tom(:, m) = interp1(gri.x, kpol_old(:, m), x_tom_mat(:, m), 'linear', 'extrap');
        end
        
        % Enforce budget feasibility only
        k_tom_tom = max(0, min(k_tom_tom, x_tom_mat - 1e-10));
        
        % Consumption tomorrow and marginal utility with stochastic R
        c_tom_mat = x_tom_mat - k_tom_tom;                   % Nkap x N_het
        MU_R_tom = repmat(R_tom, Nkap, 1) .* (c_tom_mat .^ (-par.sigma));  % Nkap x N_het
        
        % --- Step 2: Expected MU for each current state j ---
        exp_MU = MU_R_tom * gri.prob';  % Nkap x N_het
        
        % --- Step 3: Invert Euler Equation ---
        c_tod = (par.beta * exp_MU) .^ (-1/par.sigma);  % Nkap x N_het
        
        % --- Step 4: Endogenous grid ---
        x_end = gri.k + c_tod;  % Nkap x N_het
        
        % --- Step 5: Interpolate back with borrowing constraint only ---
        for j = 1:N_het
            x_end_bc = [0; x_end(:, j)];
            k_with_BC = [0; gri.k];
            
            kpol_new(:, j) = interp1(x_end_bc, k_with_BC, gri.x, 'linear', 'extrap');
            % Only borrowing constraint. NO upper-bound clamp.
            kpol_new(:, j) = max(0, kpol_new(:, j));
        end
        
        diff = max(abs(kpol_new - kpol_old), [], 'all');
        kpol_old = kpol_new;
        iter = iter + 1;
    end
    
    % Store the final x-policy
    solving_EGM.kpol_x = kpol_new;
    
    % Map back from (x, state) space to standard (k, state) space
    solving_EGM.kpol_k = zeros(Nkap, N_het);
    for j = 1:N_het
        R_today = 1 + par.r_bar + gri.r(j);
        x_today = gri.z(j) * par.w + R_today * gri.k;
        solving_EGM.kpol_k(:, j) = interp1(gri.x, kpol_new(:, j), x_today, 'linear', 'extrap');
    end
    
    % Only borrowing constraint. NO upper-bound clamp.
    solving_EGM.kpol_k = max(0, solving_EGM.kpol_k);
end