function [V, cpol, kpol] = VFI_GS(par, gri, cpar)
% VFI_GS: Solves the household problem using Value Function Iteration.
%
% VECTORIZED VERSION: Uses vectorized grid search instead of per-point
% Golden Section Search. For each productivity state j, evaluates the
% Bellman equation on all (k, k') pairs simultaneously using matrix ops,
% then refines the optimum with a local bisection step.
%
% Outputs:
%   V    : Value function matrix (N_states x N_capital)
%   cpol : Optimal consumption policy function
%   kpol : Optimal savings (next-period capital) policy function

tic;

Nz = cpar.N;
Nk = cpar.Nkap;

% Ensure column vectors
k_grid = gri.k(:);   % Nk x 1
z_grid = gri.z(:);   % Nz x 1

V = zeros(Nz, Nk);
V_new = zeros(Nz, Nk);
cpol = zeros(Nz, Nk);
kpol = zeros(Nz, Nk);

% Build a fine search grid for k' that covers the FULL feasible range.
% The maximum feasible k' is R * k_max + w * z_max (the full budget).
kp_max = par.R * max(k_grid) + par.w * max(z_grid);

% If extrapolation is disabled, cap at the grid max
if isfield(cpar, 'extrapolate') && cpar.extrapolate == false
    kp_max = max(k_grid);
end

% Use 200 points for high accuracy (comparable to Golden Section precision)
Nk_fine = 200;
kp_fine = linspace(0, kp_max, Nk_fine)';  % Nk_fine x 1

iter = 0;
diff = cpar.tol + 1;
cpar.maxit = 1500;

fprintf('Starting Vectorized VFI...\n');

while iter < cpar.maxit && diff > cpar.tol
    iter = iter + 1;
    
    % Pre-interpolate continuation values on the fine k' grid for ALL states
    % Use 'extrap' to handle k' values beyond the solved grid
    V_fine = zeros(Nz, Nk_fine);
    for jp = 1:Nz
        V_fine(jp, :) = interp1(k_grid, V(jp, :), kp_fine, 'linear', 'extrap');
    end
    
    % Expected continuation: EV(j, kp_idx) = sum_jp P(j, jp) * V_fine(jp, kp_idx)
    EV = gri.prob * V_fine;  % Nz x Nk_fine
    
    for j = 1:Nz
        z_j = z_grid(j);
        
        % Cash-on-hand for each current k: Nk x 1
        coh = par.R * k_grid + par.w * z_j;
        
        % Consumption matrix: c(i, kp_idx) = coh(i) - kp_fine(kp_idx)
        % Size: Nk x Nk_fine
        C_mat = coh - kp_fine';
        
        % Utility: penalize infeasible choices (negative consumption or negative k')
        U_mat = -1e10 * ones(Nk, Nk_fine);
        feasible = C_mat > 1e-10;
        U_mat(feasible) = C_mat(feasible).^(1-par.sigma) / (1-par.sigma);
        
        % Total value: u(c) + beta * E[V(k', z')]
        % EV(j, :) is 1 x Nk_fine, broadcast over Nk rows
        Total = U_mat + par.beta * repmat(EV(j, :), Nk, 1);
        
        % Find optimal k' for each current k
        [V_new(j, :), idx_opt] = max(Total, [], 2);
        kpol(j, :) = kp_fine(idx_opt)';
        cpol(j, :) = coh' - kpol(j, :);
        
        % === LOCAL REFINEMENT via Golden Section on the found interval ===
        % Refine each (j,i) point within [kp_fine(idx-1), kp_fine(idx+1)]
        gr = (sqrt(5) - 1) / 2;
        for i = 1:Nk
            lo = max(0, kp_fine(max(idx_opt(i)-1, 1)));
            hi = min(coh(i) - 1e-10, kp_fine(min(idx_opt(i)+1, Nk_fine)));
            if hi <= lo; continue; end
            
            % 10 bisection-style refinement steps
            for ref = 1:10
                c_ref = hi - gr * (hi - lo);
                d_ref = lo + gr * (hi - lo);
                
                % Evaluate Bellman at c_ref
                C_c = coh(i) - c_ref;
                if C_c > 1e-10
                    val_c = C_c^(1-par.sigma)/(1-par.sigma) + par.beta * interp1(kp_fine, EV(j,:), c_ref, 'linear', 'extrap');
                else
                    val_c = -1e10;
                end
                
                % Evaluate Bellman at d_ref
                C_d = coh(i) - d_ref;
                if C_d > 1e-10
                    val_d = C_d^(1-par.sigma)/(1-par.sigma) + par.beta * interp1(kp_fine, EV(j,:), d_ref, 'linear', 'extrap');
                else
                    val_d = -1e10;
                end
                
                if val_c > val_d
                    hi = d_ref;
                else
                    lo = c_ref;
                end
            end
            
            % Best point
            if val_c > val_d
                kpol(j,i) = c_ref;
                V_new(j,i) = val_c;
            else
                kpol(j,i) = d_ref;
                V_new(j,i) = val_d;
            end
            cpol(j,i) = coh(i) - kpol(j,i);
        end
        
        % === EXTRAPOLATION TOGGLE ===
        if isfield(cpar, 'extrapolate') && cpar.extrapolate == false
            kpol(j, :) = min(kpol(j, :), max(k_grid));
            cpol(j, :) = coh' - kpol(j, :);
        end
    end
    
    % Convergence check
    diff = max(abs(V_new(:) - V(:)));
    V = V_new;

    if iter <= 5 || mod(iter, 10) == 0
        fprintf('Iter %d: diff = %.8e, V_min = %.4f, V_max = %.4f\n', ...
                iter, diff, min(V_new(:)), max(V_new(:)));
    end
end

vfi_time = toc;

if diff <= cpar.tol
    fprintf('VFI converged after %d iterations (sup norm: %.8f)\n', iter, diff);
else
    fprintf('VFI did not converge after %d iterations (sup norm: %.8f)\n', cpar.maxit, diff);
end

fprintf('VFI execution time: %.2f seconds\n', vfi_time);
if vfi_time > 0
    fprintf('Average iteration speed: %.2f iterations/second\n', iter/vfi_time);
end

end