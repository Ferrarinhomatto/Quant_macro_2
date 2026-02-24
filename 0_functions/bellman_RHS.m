function bellman = bellman_RHS(C, K, Z, par, cpar, gri, j, V)
% 
% % INPUTS:
% C: guess for consumption
% K: the current state of K
% A: the current state of A
% par, cpar: the parameters given in the model
% interEV: function for getting the expected value
% 
% get N given the guess on consumption and the rest of the variables

if C <= 1e-10
    bellman = - 1e10; % penalities
    return
end



k_prime = k_motion(K, Z, C, par);

if k_prime < 0;

    bellman = -1e10;
    
    return
end

% Feasibility constraints are finished

exp_value = 0;

for j_prime = 1:cpar.N

    % interpolate V at (k_prime, a_prime)

    V_inter = interp_kp(gri.k, V(j_prime, :), k_prime);

    exp_value = exp_value + gri.prob(j, j_prime) * V_inter;
end

% Define the RHS

    total = u(C, par) + par.beta * exp_value;

    bellman = total;

end



