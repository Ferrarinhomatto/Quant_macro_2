function V_kprime = interp_kp(gri_k, V_k, k_prime)

% This function serves for doing interpolation on the grid of capital, at
% points of k_prime implied by the consumption guesses

% differently from last weeek, we need extrapolation to be done on both
% sides

% First, we handle scalar or vector inputs
if isscalar(k_prime)
    xq_vec = k_prime;
else
    xq_vec = k_prime(:);
end

n = length(xq_vec);
V_kprime = zeros(size(xq_vec));

for k = 1:n
    k_prime = xq_vec(k);

    % handle extrapolation cases
    if k_prime >= gri_k(end) 
        if k_prime == gri_k(end) 
            V_kprime(k) = V_k(end);
        else
        slope = (V_k(end) - V_k(end-1)) / (gri_k(end)- gri_k(end-1));
        V_kprime(k) = V_k(end) + slope * (k_prime - gri_k(end));
        end
    elseif k_prime <= gri_k(1) 
        if k_prime == gri_k(1)
           V_kprime(k) = V_k(1);
         elseif k_prime <= 1e-6
           V_kprime(k) = -1e10;
        else    
        slope = (V_k(2) - V_k(1)) / (gri_k(2) - gri_k(1));
        V_kprime(k) = V_k(1) - slope * (gri_k(1) - k_prime);
        end
    else
        idx = find(gri_k <= k_prime, 1,  'last');

        % Linear interpolation formula

        weight = (k_prime - gri_k(idx))/ (gri_k(idx+1) - gri_k(idx));
        V_kprime(k) = V_k(idx) + weight * (V_k(idx + 1) - V_k(idx));
    end 
end