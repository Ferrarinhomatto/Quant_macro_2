function solving_EGM = EGM(par, cpar, gri)
% EGM: Solves the household problem using Carroll's Endogenous-Gridpoints Method.
%
% The EGM algorithm works backwards. Instead of iterating over today's cash-on-hand 
% to find optimal savings, it pre-defines a grid for savings (gri.k), calculates expected 
% marginal utility tomorrow, derives consumption today from the Euler Equation, 
% and finally constructs the endogenous cash-on-hand grid today.

% Store the grids we had before

gri.k = gri.k;
gri.z = gri.z;

% Define endogenous grid space bounds for cash-on-hand (x)
x_min = min(gri.z) * par.w  + par.R * gri.k(1);
x_max = max(gri.z) * par.w + par.R * gri.k(end);

gri.x = linspace(x_min, x_max, 100);

kpol_old = zeros(length(gri.x), cpar.N);


% Start loop

for j = 1:cpar.N
    
    kpol_old(:, j) = 0.5 * gri.x';

end

diff = cpar.tol + 1;
cpar.maxit = 1000;
iter = 0;

while diff > cpar.tol && iter < cpar.maxit
    % Update policy function and calculate the difference
    kpol_new = zeros(size(kpol_old));

    for j = 1:cpar.N

        c_tod = zeros(cpar.Nkap, 1);
        x_end = zeros(cpar.Nkap, 1);

        for kp = 1:cpar.Nkap

            exp_MU = 0;

            for z = 1:cpar.N

                x_tom = gri.z(z) * par.w + par.R * gri.k(kp);

                k_tom_tom = interp1(gri.x, kpol_old(:,z), x_tom, 'linear', 'extrap');

                k_tom_tom = max(0, min(k_tom_tom, x_tom - 1e-10));

                c_tom = x_tom - k_tom_tom;

                % Tomorrow's probability * MU(c_tomorrow)
                prob_transition = gri.prob(j, z);
                exp_MU = exp_MU + prob_transition * (c_tom^(-par.sigma));

            end

            % Invert the Euler Equation to find implied consumption TODAY
            c_tod(kp) = (par.beta * par.R *exp_MU)^(-1/par.sigma);
            
            % The Endogenous Gridpoint (today's required cash on hand): x_today = k_tomorrow + c_today
            x_end(kp) = gri.k(kp) + c_tod(kp);

        end

        % Borrowing constraint
        
        x_end_bc = [0; x_end];

        if isrow(gri.k)
            k_with_BC = [0; gri.k']; 
        else
            k_with_BC = [0; gri.k];
        end
        
        kpol_new(:, j) = interp1(x_end_bc, k_with_BC, gri.x, "linear","extrap");

        kpol_new(:, j) = max(0, min(kpol_new(:, j), gri.x'));
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

% We can also easily recover the consumption policy on the x grid
% Because x = c + k'  =>  c = x - k'
solving_EGM.cpol_x = repmat(gri.x', 1, cpar.N) - kpol_new;

% To make this comparable with VFI from Question 1.A, we map the policy back to the (k, z) state space instead of (x, z). 
solving_EGM.kpol_k = zeros(cpar.Nkap, cpar.N);
solving_EGM.cpol_k = zeros(cpar.Nkap, cpar.N);

for j = 1:cpar.N
    % Cash-on-hand for each k grid point today [cite: 38]
    x_today = gri.z(j) * par.w + par.R * gri.k; 
    
    % Interpolate the x-policies onto the k-grid
    solving_EGM.kpol_k(:, j) = interp1(gri.x, solving_EGM.kpol_x(:, j), x_today, 'linear', 'extrap')';
    solving_EGM.cpol_k(:, j) = interp1(gri.x, solving_EGM.cpol_x(:, j), x_today, 'linear', 'extrap')';
end

end