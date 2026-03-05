function solving_EGM = EGM(par, cpar, gri)
% EGM: Solves the household problem using Carroll's Endogenous-Gridpoints Method.
%
% VECTORIZED VERSION with:
%   - Log-spaced cash-on-hand grid (matches the log-spaced k grid for
%     better resolution near the borrowing constraint)
%   - No artificial upper-bound clamping on k' (only borrowing constraint enforced)

gri.k = gri.k(:);   % Ensure column
gri.z = gri.z(:);

% Define endogenous grid space bounds for cash-on-hand (x)
x_min = min(gri.z) * par.w  + par.R * gri.k(1);
x_max = max(gri.z) * par.w + par.R * gri.k(end);

% Log-spaced x grid: concentrate points near zero where the borrowing
% constraint binds and the policy function has the most curvature.
% Same log-transform as the k grid: x = exp(linspace(log(x_min+1), log(x_max+1), Nx)) - 1
Nx = 200;
gri.x = exp(linspace(log(x_min + 1), log(x_max + 1), Nx))' - 1;

% Initial guess: save half of cash-on-hand
kpol_old = 0.5 * repmat(gri.x, 1, cpar.N);

diff = cpar.tol + 1;
cpar.maxit = 1000;
iter = 0;

% Pre-compute tomorrow's cash-on-hand for each (kp, z') pair: Nkap x N matrix
% x_tom(kp, z') = z' * w + R * k(kp)
x_tom_mat = par.R * gri.k + (par.w * gri.z');  % Nkap x N

while diff > cpar.tol && iter < cpar.maxit
    
    % --- Step 1: Batch-interpolate tomorrow's policy for all future states ---
    k_tom_tom = zeros(cpar.Nkap, cpar.N);
    for z = 1:cpar.N
        k_tom_tom(:, z) = interp1(gri.x, kpol_old(:, z), x_tom_mat(:, z), 'linear', 'extrap');
    end
    
    % Enforce constraints: 0 <= k'' <= x_tom - epsilon (budget feasibility only)
    k_tom_tom = max(0, min(k_tom_tom, x_tom_mat - 1e-10));
    
    % Consumption tomorrow: c' = x' - k''
    c_tom_mat = x_tom_mat - k_tom_tom;  % Nkap x N
    
    % Marginal utility tomorrow: u'(c') = c'^(-sigma)
    MU_tom = c_tom_mat .^ (-par.sigma);  % Nkap x N
    
    % --- Step 2: Compute expected MU for each current state j ---
    exp_MU = MU_tom * gri.prob';  % Nkap x N
    
    % --- Step 3: Invert Euler Equation to get consumption today ---
    c_tod = (par.beta * par.R * exp_MU) .^ (-1/par.sigma);  % Nkap x N
    
    % --- Step 4: Endogenous grid point: x_today = k' + c_today ---
    x_end = gri.k + c_tod;  % Nkap x N
    
    % --- Step 5: Interpolate back onto the x grid with borrowing constraint ---
    kpol_new = zeros(length(gri.x), cpar.N);
    for j = 1:cpar.N
        x_end_bc = [0; x_end(:, j)];
        k_with_BC = [0; gri.k];
        
        kpol_new(:, j) = interp1(x_end_bc, k_with_BC, gri.x, 'linear', 'extrap');
        % Only enforce borrowing constraint (k' >= 0). NO upper-bound clamp.
        kpol_new(:, j) = max(0, kpol_new(:, j));
    end
    
    diff = max(abs(kpol_new - kpol_old), [], 'all');
    kpol_old = kpol_new;
    iter = iter + 1;

    if mod(iter, 50) == 0
        fprintf('EGM Iteration: %d, Diff: %e\n', iter, diff);
    end
end

fprintf('EGM converged successfully in %d iterations!\n', iter);

% Store the final policy in the output struct
solving_EGM.kpol_x = kpol_new;

% Consumption policy on the x grid: c = x - k'
solving_EGM.cpol_x = repmat(gri.x, 1, cpar.N) - kpol_new;

% Map back to the (k, z) state space
solving_EGM.kpol_k = zeros(cpar.Nkap, cpar.N);
solving_EGM.cpol_k = zeros(cpar.Nkap, cpar.N);

for j = 1:cpar.N
    x_today = gri.z(j) * par.w + par.R * gri.k; 
    solving_EGM.kpol_k(:, j) = interp1(gri.x, solving_EGM.kpol_x(:, j), x_today, 'linear', 'extrap');
    solving_EGM.cpol_k(:, j) = interp1(gri.x, solving_EGM.cpol_x(:, j), x_today, 'linear', 'extrap');
end

% Only enforce borrowing constraint on final output. NO upper-bound clamp.
solving_EGM.kpol_k = max(0, solving_EGM.kpol_k);

end