function err = euler_errors(beta, sigma, R, w, policy_k, kgrid, zgrid, P)

[Nz, Nk] = size(policy_k); 
err = zeros(Nz, Nk);

for iz = 1:Nz
    
    % --- 1. Current State (k, z)
    k_now = kgrid(:)';          % 1xNk
    z_now = zgrid(iz);         % scalar
    kp    = policy_k(iz, :);   % 1xNk
    
    % Current consumption
    c_now = R * k_now + w * z_now - kp;
    c_now(c_now <= 1e-12) = NaN;  
    
    % --- 2. Expected Marginal Utility Tomorrow
    expected_MU_next = zeros(1, Nk);
    
    for izp = 1:Nz
        
        z_next = zgrid(izp);
        
        % Interpolate future capital choice g(k',z')
        k_pp = interp1(kgrid, policy_k(izp,:), kp, 'linear', 'extrap');
        k_pp = max(k_pp, 0);
        
        % c' = R*k' + w*z' - k''
        c_next = R * kp + w * z_next - k_pp;
        c_next = max(c_next, 1e-12);
        
        % Contribution to expected MU
        expected_MU_next = expected_MU_next + ...
            P(iz, izp) * beta * R * (c_next.^(-sigma));
    end
    
    % Implied consumption from Euler equation
    c_implied = expected_MU_next.^(-1/sigma);
    
    % Raw Euler error
    ee = (c_implied - c_now) ./ c_now;
    
    % --- 3. Enforce inequality Euler condition at binding points ---
    binding = abs(kp - kgrid(1)) < 1e-5;   % borrowing constraint binds
    
    % When constraint binds, inequality is u'(c) >= beta R E[u'(c')]
    % If inequality holds, there is NO Euler equation error.
    ee(binding) = min(ee(binding), 0);  % if positive (valid slack), set to 0. 
    
    err(iz, :) = abs(ee) * 100; % Return purely absolute percentages
end
end