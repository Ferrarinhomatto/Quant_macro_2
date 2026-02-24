function V_kprime = interp_kp(gri_k, V_k, k_prime)

% This function serves for doing interpolation on the grid of capital, at
% points of k_prime implied by the consumption guesses

% First, we handle scalar or vector inputs
if isscalar(k_prime)
    xq_vec = k_prime;
else
    xq_vec = k_prime(:);
end

n = length(xq_vec);
V_kprime = zeros(size(xq_vec));

for k = 1:n
    kp = xq_vec(k);

    % handle extrapolation cases
    if kp >= gri_k(end) 
        slope = (V_k(end) - V_k(end-1)) / (gri_k(end)- gri_k(end-1));
        V_kprime(k) = V_k(end) + slope * (kp - gri_k(end));
    elseif kp <= gri_k(1) 
        slope = (V_k(2) - V_k(1)) / (gri_k(2) - gri_k(1));
        V_kprime(k) = V_k(1) + slope * (kp - gri_k(1));
    else
        idx = find(gri_k <= kp, 1,  'last');

        % Linear interpolation formula
        weight = (kp - gri_k(idx))/ (gri_k(idx+1) - gri_k(idx));
        V_kprime(k) = V_k(idx) + weight * (V_k(idx + 1) - V_k(idx));
    end 
end
end
