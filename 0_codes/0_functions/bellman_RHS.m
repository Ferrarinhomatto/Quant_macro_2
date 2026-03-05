function bellman = bellman_RHS_2(k_prime, K, Z, par, cpar, gri, j, V)

% Compute consumption from the budget constraint
% c_t + k_{t+1} = R*k_t + z_t*w
C = par.R * K + par.w * Z - k_prime;

if C <= 1e-10
    bellman = -1e10; % Penalty to ensure Golden Section Search avoids negative/zero consumption
    return
end

if k_prime < -1e-8
    bellman = -1e10; % Penalty to strictly enforce the borrowing limit (k' >= 0, or par.b)
    return
end

% === EXTRAPOLATION TOGGLE ===
% If we disabled extrapolation, physically cap the agent's expected continuation value evaluation
if isfield(cpar, 'extrapolate') && cpar.extrapolate == false
    k_prime_eval = min(k_prime, max(gri.k));
else
    k_prime_eval = k_prime;
end

exp_value = 0;

for j_prime = 1:cpar.N
    % ALWAYS use 'extrap' here to prevent MATLAB crashes.
    V_inter = interp1(gri.k, V(j_prime, :), k_prime_eval, 'linear', 'extrap');
    exp_value = exp_value + gri.prob(j, j_prime) * V_inter;
end

% Define the RHS
bellman = u(C, par) + par.beta * exp_value;

end
