function bellman = bellman_RHS(C, K, Z, par, cpar, gri, j, V)

if C <= 1e-10
    bellman = - 1e10; % Penalty to ensure Golden Section Search avoids negative/zero consumption
    return
end

k_prime = k_motion(K, Z, C, par);

if k_prime < 1e-8
    bellman = -1e10; % Penalty to strictly enforce the borrowing limit (k' >= 0)
    return
end

% === EXTRAPOLATION TOGGLE ===
% If we disabled extrapolation, physically cap the agent's savings at the grid max
if isfield(cpar, 'extrapolate') && cpar.extrapolate == false
    k_prime = min(k_prime, max(gri.k));
end

exp_value = 0;

for j_prime = 1:cpar.N
    % ALWAYS use 'extrap' here to prevent MATLAB crashes.
    % If cpar.extrapolate == false, k_prime is already capped above, so 
    % it won't actually extrapolate anyway!
    V_inter = interp1(gri.k, V(j_prime, :), k_prime, 'linear', 'extrap');
    exp_value = exp_value + gri.prob(j, j_prime) * V_inter;
end

% Define the RHS
bellman = u(C, par) + par.beta * exp_value;

end